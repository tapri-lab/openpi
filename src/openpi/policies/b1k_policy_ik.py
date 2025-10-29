import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, ACTION_QPOS_INDICES
from omnigibson.utils.transform_utils import quat_multiply, quat_inverse


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4

    eef_left_pos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_pos"]] # 3
    eef_left_quat = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_quat"]] # 4
    eef_right_pos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_pos"]] # 3
    eef_right_quat = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_quat"]] # 4

    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        eef_left_pos,
        eef_left_quat,
        eef_right_pos,
        eef_right_quat,
        left_gripper_width,
        right_gripper_width,
    ], axis=-1)

def ang_delta_from_quat(quaternions):
    dq = np.diff(quaternions, axis=0)
    q_inv = quat_inverse(quaternions[:-1])
    ang_delta = 2 * quat_multiply(q_inv, dq.T)

    return ang_delta


# def extract_action(data):
#     proprio_data = data["observation/state"]
#     eef_left_lin_vel = np.vstack((np.diff(proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_pos"]], axis=0), np.zeros((1,3))))
#     eef_right_lin_vel = np.vstack((np.diff(proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_pos"]], axis=0), np.zeros((1,3))))
# 
#     eef_left_ang_vel = np.vstack((ang_delta_from_quat(proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_quat"]]), np.zeros((1,3))))
#     eef_right_ang_vel = np.vstack((ang_delta_from_quat(proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_quat"]]), np.zeros((1,3))))
# 
#     return np.concatenate([
#         data["action"][..., ACTION_QPOS_INDICES["R1Pro"]["base"]],
#         data["action"][..., ACTION_QPOS_INDICES["R1Pro"]["torso"]],
#         eef_left_lin_vel,
#         eef_left_ang_vel,
#         data["action"][..., ACTION_QPOS_INDICES["R1Pro"]["left_gripper"]],
#         eef_right_lin_vel,
#         eef_right_ang_vel,
#         data["action"][..., ACTION_QPOS_INDICES["R1Pro"]["right_gripper"]],
#     ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action =  data["actions"][:, :21]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 21
    def __call__(self, data: dict) -> dict:
        # Only return the first 21 dims.
        actions = data["actions"]
        return {"actions": np.asarray(actions[:, :self.action_dim])}
