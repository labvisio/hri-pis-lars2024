from easydict import EasyDict

from is_3d_skeleton.backend.PoseTracker3D.pose_tracker_3d import PoseTracker3D
from tracking.IterativeTracker import IterativeTracker


class Skeleton3D:

    def __init__(self, cameras, service_cfg):
        self._person_det_cfg = None
        self._pose_det_cfg = None
        self._tracker_cfg = None
        self._conf_threshold = None
        self._build_3d = None
        self._cameras = cameras

        self._load_conf(service_cfg)

    def _load_conf(self, cfg):
        pipeline = cfg.PIPELINE_COMBINATION
        self._tracker_cfg = cfg.PERSON_MATCHERS[pipeline['PERSON_MATCHER'].upper()]
        self._conf_threshold = pipeline['CONF_THRESHOLD']
        self._build_3d = pipeline['BUILD_3D']
        self._pose_tracker_3d = self._get_pose_tracker_3d()

    def _get_pose_tracker_3d(self):
        assert self._tracker_cfg is not None

        if self._tracker_cfg.NAME == 'Iterative':
            # TODO: modificar isso depois. Tá muito feio!!
            iterative_tracker_args = EasyDict()
            iterative_tracker_args.conf_threshold = self._conf_threshold
            iterative_tracker_args.epi_threshold = self._tracker_cfg.EPI_THRESHOLD
            iterative_tracker_args.init_threshold = self._tracker_cfg.INIT_THRESHOLD
            iterative_tracker_args.joint_threshold = self._tracker_cfg.JOINT_THRESHOLD
            iterative_tracker_args.num_positive_affinities = self._tracker_cfg.NUM_POSITIVE_AFFINITIES
            iterative_tracker_args.num_joints = self._tracker_cfg.NUM_JOINTS
            iterative_tracker_args.init_method = self._tracker_cfg.INIT_METHOD
            iterative_tracker_args.n_init = self._tracker_cfg.N_INIT
            iterative_tracker_args.max_age = self._tracker_cfg.MAX_AGE
            iterative_tracker_args.w2d = self._tracker_cfg.W2D
            iterative_tracker_args.alpha2d = self._tracker_cfg.ALPHA2D
            iterative_tracker_args.w3d = self._tracker_cfg.W3D
            iterative_tracker_args.alpha3d = self._tracker_cfg.ALPHA3D
            iterative_tracker_args.lambda_a = self._tracker_cfg.LAMBDA_A
            iterative_tracker_args.lambda_t = self._tracker_cfg.LAMBDA_T
            iterative_tracker_args.sigma = self._tracker_cfg.SIGMA
            iterative_tracker_args.arm_sigma = self._tracker_cfg.ARM_SIGMA

            tracker = IterativeTracker(iterative_tracker_args)
        else:
            raise NotImplementedError(f"{self._tracker_cfg.NAME} Person Matcher has not yet been implemented.")

        pose_tracker_3d = PoseTracker3D(tracker, self._cameras, self._build_3d)
        return pose_tracker_3d

    def estimate(self, frame_id, obs2d_dict, result_pose):
        return self._pose_tracker_3d(frame_id, obs2d_dict, result_pose)
