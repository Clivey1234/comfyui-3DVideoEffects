from .latent_sbs import LatentStereoCamera
from .smallest_size import SmallestSize
from .frame_progress import FrameProgress
from .vhs_path_cleaner import VHSPathCleaner
from .image_out_selector import ImageOutSelector
from .image_in_selector import ImageInSelector
from .depth_anything_spb import DepthAnything2Image
from .leres_spb import LeReSDepth2Image
from .depth_multi_spb import DepthMultiBackend2Image
from .image_keyframes import ImageKeyFrames
from .vfi_ease_ends import VFIEaseEnds
from .vfi_bullet_time_markers import VFIBulletTimeMarkers
from .Pop3D import Pop3DClass
from .pulfrich_sbs_from_sequence import PulfrichSBSFromSequence
from .anaglyph_3d import Anaglyph3D
from .youtube_3d_metadata_sbs import YouTube3DMetadataSBSLR
from .batch_pingpong import BatchPingPong
from .text_overlay_timeline import TextOverlayTimeline
from .batch_number_overlay import ImageBatchNumberOverlay

NODE_CLASS_MAPPINGS = {
    "LatentStereoCamera": LatentStereoCamera,
    "ImageResInfo": SmallestSize,
    "FrameProgress": FrameProgress,
    "VHSPathCleaner": VHSPathCleaner,
    "ImageOutSelector": ImageOutSelector,
    "ImageInSelector": ImageInSelector,
    "DepthAnything": DepthAnything2Image,
    "LeReSDepth": LeReSDepth2Image,
    "DepthMultiDepth": DepthMultiBackend2Image,
    "ImageKeyFrames": ImageKeyFrames,
    "VFIEaseEnds": VFIEaseEnds,
    "VFIBulletTimeMarkers": VFIBulletTimeMarkers,
    "Pop3DClass": Pop3DClass,
    "PulfrichSBSFromSequence": PulfrichSBSFromSequence,
    "Anaglyph3D": Anaglyph3D,
    "YouTube3DMetadataSBSLR": YouTube3DMetadataSBSLR,
    "BatchPingPong": BatchPingPong,
    "TextOverlayTimeline": TextOverlayTimeline,
    "ImageBatchNumberOverlay": ImageBatchNumberOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentStereoCamera": "3DVidTools_SBSCamera🟢📺",
    "ImageResInfo": "3DVidTools_ImageResInfo🟢🖼️",
    "FrameProgress": "3DVidTools_Frame Progress🟢📈",
    "VHSPathCleaner": "3DVidTools_VHS Path Cleaner🟢🛁",
    "ImageOutSelector": "3DVidTools_Image Out Selector🟢🖼️",
    "ImageInSelector": "3DVidTools_Image In Selector🟢🖼️",
    "DepthAnything": "3DVidTools_Depth Anything (2-Image)🟢📺",
    "LeReSDepth": "3DVidTools_LeReS Depth (2-Image)🟢📺",
    "DepthMultiDepth": "3DVidTools_Depth Anything / LeReS🟢📺",
    "ImageKeyFrames": "3DVidTools_Image Key Frames🟢🎥",
    "VFIEaseEnds": "3DVidTools_VFI Ease Ends🟢🎥",
    "VFIBulletTimeMarkers": "3DVidTools_VFI Bullet Time Markers🟢🎥",
    "Pop3DClass": "3DVidTools_3DPop🟢🎥",
    "PulfrichSBSFromSequence": "3DVidTools_Pulfrich🟢🎥",
    "Anaglyph3D": "3DVidTools_Anaglyph3D🟢🎥",
    "YouTube3DMetadataSBSLR": "3DVidTools_YouTube3D_Metadata🟢🎥",
    "BatchPingPong": "3DVidTools_BatchPingPong🟢🎥",
    "TextOverlayTimeline": "3DVidTools_TextOverlayTimeline🟢🎥",
    "ImageBatchNumberOverlay": "3DVidTools_ImageBatchNumberOverlay🟢🎥",
}
