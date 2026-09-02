import os
import sys 
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from safetensors.torch import load_file
from insightface.app import FaceAnalysis
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.parsing import init_parsing_model

sys.path.insert(0, os.path.expanduser("~/PuLID-FLUX"))
from eva_clip import create_model_and_transforms

from pulid.encoders_transformer import IDFormer

PULID_PATH      = os.path.expanduser("~/pulid_weights/pulid_flux_v0.9.1.safetensors")
INSIGHTFACE_DIR = os.path.expanduser("~/insightface")
FACE_APP_PROVIDER = "CUDAExecutionProvider"

class FaceProcessor:
    
    

    def __init__(self, DEVICE, DTYPE):

        self.DEVICE = DEVICE
        self.DTYPE = DTYPE

        self._file_embeddings: dict[str, torch.Tensor] = {}
        self._file_embeddings_max = 8

        print("[FaceProcessor] Loading EVA CLIP...")
        eva_clip, _, _ = create_model_and_transforms(
            "EVA02-CLIP-L-14-336",
            os.path.expanduser("~/eva_clip_weights/EVA02_CLIP_L_336_psz14_s6B.pt"),
            force_custom_clip=True,
        )

        self.eva_clip = eva_clip.visual.to(DEVICE, dtype=torch.float32).eval()

        for block in self.eva_clip.blocks:
            block.attn.xattn = False

        print("[FaceProcessor] Loading InsightFace...")
        self.face_app = FaceAnalysis(
            name="antelopev2",
            root=INSIGHTFACE_DIR,
            providers=[FACE_APP_PROVIDER],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        print("[FaceProcessor] Loading PuLID IDFormer...")
        pulid_state = load_file(PULID_PATH)
        pulid_encoder_state = {
            k[len("pulid_encoder."):]: v
            for k, v in pulid_state.items()
            if k.startswith("pulid_encoder.")
        }

        self.id_encoder = IDFormer().to(DEVICE, dtype=DTYPE)
        self.id_encoder.load_state_dict(pulid_encoder_state, strict=True)
        self.id_encoder.eval()

        print("[FaceProcessor] Loading face parser...")
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=torch.device(DEVICE),
        )

        self.face_helper.face_parse = init_parsing_model(
            model_name="bisenet",
            device=torch.device(DEVICE),
        )

        self.bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        print("[FaceProcessor] Ready.")


    def image_to_tensor(self, image):
       tensor = torch.clamp(torch.from_numpy(image).float() / 255.0, 0, 1)
       tensor = tensor[..., [2, 1, 0]]
       return tensor

    def to_gray(self, img):
       x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
       return x.repeat(1, 3, 1, 1)

    def process(self, id_image: np.ndarray, filehash_cache_key: str = None) -> torch.Tensor:

        assert filehash_cache_key is not None, "No reference file hash was provided from client"

        if filehash_cache_key in  self._file_embeddings :
            print("Re-using cached embeddings")
            return self._file_embeddings[filehash_cache_key]

        print("[FaceProcessor] Processing face...")

        with torch.inference_mode():
            faces = self.face_app.get(id_image)
            assert len(faces) > 0, "No face detected in ID image"

            face = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )[0]

            id_ante_embedding = (
                torch.from_numpy(face.embedding)
                .unsqueeze(0)
                .to(self.DEVICE, dtype=self.DTYPE)
            )

            self.face_helper.clean_all()
            self.face_helper.read_image(id_image[..., ::-1].copy())
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()

            assert len(self.face_helper.cropped_faces) > 0, "No aligned face detected"

            align_face = self.face_helper.cropped_faces[0]

            align_face = (
                self.image_to_tensor(align_face)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
                .to(self.DEVICE, dtype=torch.float32)
            )

            parsing_out = self.face_helper.face_parse(
                TF.normalize(
                    align_face,
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )
            )[0]

            parsing_out = parsing_out.argmax(dim=1, keepdim=True)

            bg = sum(parsing_out == i for i in self.bg_label).bool()

            white_image = torch.ones_like(align_face)
            face_features_image = torch.where(
                bg,
                white_image,
                self.to_gray(align_face),
            )

            face_features_image = TF.resize(
                face_features_image,
                self.eva_clip.image_size,
                transforms.InterpolationMode.BICUBIC,
            ).to(self.DEVICE, dtype=torch.float32)

            face_features_image = TF.normalize(
                face_features_image,
                self.eva_clip.image_mean,
                self.eva_clip.image_std,
            )

            print("[FaceProcessor] Extracting EVA face features...")

            id_cond_vit, id_vit_hidden = self.eva_clip(
                face_features_image,
                return_all_features=False,
                return_hidden=True,
                shuffle=False,
            )

            id_cond_vit = id_cond_vit.to(self.DEVICE, dtype=self.DTYPE)
            id_vit_hidden = [x.to(self.DEVICE, dtype=self.DTYPE) for x in id_vit_hidden]

            id_cond_vit = torch.div(
                id_cond_vit,
                torch.norm(id_cond_vit, 2, 1, True),
            )

            id_cond = torch.cat(
                [id_ante_embedding, id_cond_vit],
                dim=-1,
            )

            print("[FaceProcessor] Creating PuLID id_embeddings...")

            id_embeddings = self.id_encoder(id_cond, id_vit_hidden)

            print("id_embeddings shape:", id_embeddings.shape)
            print("id_embeddings mean abs:", id_embeddings.abs().mean().item())
            print("id_embeddings min/max:", id_embeddings.min().item(), id_embeddings.max().item())
          
            if len(self._file_embeddings) >= self._file_embeddings_max:
                oldest = next(iter(self._file_embeddings))
                del self._file_embeddings[oldest]
            self._file_embeddings[filehash_cache_key] = id_embeddings

        return id_embeddings