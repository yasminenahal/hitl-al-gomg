import os
import json
from reinvent_scoring.configs.config import reinvent_scoring_config as config


project_root = os.path.dirname(__file__)

MAIN_TEST_PATH = config["MAIN_TEST_PATH"]
ACTIVITY_REGRESSION = "unittest_reinvent/fixtures/dummy_regression_model.pkl"
ACTIVITY_CLASSIFICATION = "unittest_reinvent/fixtures/dummy_classification_model.pkl"
SAS_MODEL_PATH = "/projects/mai/software/reinvent_data/non_user/test_data/predictive_models/SA_score_prediction.pkl"

# component specific
AZDOCK_ENV_PATH = config["COMPONENT_SPECIFIC"]["AZDOCK"]["AZDOCK_ENV_PATH"]
AZDOCK_DOCKER_SCRIPT_PATH = config["COMPONENT_SPECIFIC"]["AZDOCK"]["AZDOCK_DOCKER_SCRIPT_PATH"]
AZDOCK_DEBUG = config["COMPONENT_SPECIFIC"]["AZDOCK"]["AZDOCK_DEBUG"]
AZDOCK_UNITTEST_JSON = project_root + "/../scoring_tests/fixtures/azdock_data/azdock_OpenEye.json"
AZDOCK_UNITTEST_OE_RECEPTOR_PATH = project_root + "/../scoring_tests/fixtures/azdock_data/azdock_OpenEye_receptor.oeb"

DOCKSTREAM_ENV_PATH = config["COMPONENT_SPECIFIC"]["DOCKSTREAM"]["DOCKSTREAM_ENV_PATH"]
DOCKSTREAM_DOCKER_SCRIPT_PATH = config["COMPONENT_SPECIFIC"]["DOCKSTREAM"]["DOCKSTREAM_DOCKER_SCRIPT_PATH"]
DOCKSTREAM_DEBUG = config["COMPONENT_SPECIFIC"]["DOCKSTREAM"]["DOCKSTREAM_DEBUG"]
DOCKSTREAM_UNITTEST_JSON = project_root + "/../scoring_tests/fixtures/dockstream_data/dockstream_OpenEye.json"
DOCKSTREAM_UNITTEST_OE_RECEPTOR_PATH = project_root + "/../scoring_tests/fixtures/dockstream_data/dockstream_OpenEye_receptor.oeb"

ICOLOS_EXECUTOR_PATH = config["COMPONENT_SPECIFIC"]["ICOLOS"]["ICOLOS_EXECUTOR_PATH"]
ICOLOS_DEBUG = config["COMPONENT_SPECIFIC"]["ICOLOS"]["ICOLOS_DEBUG"]
ICOLOS_UNITTEST_JSON = project_root + "/../scoring_tests/fixtures/icolos_data/icolos_NIBR.json"
ICOLOS_UNITTEST_GRID_PATH = project_root + "/../scoring_tests/fixtures/icolos_data/icolos_cox2_grid.zip"
ICOLOS_UNITTEST_NIBR_NEGATIVE_IMAGE = project_root + "/../scoring_tests/fixtures/icolos_data/icolos_NIBR_negative_image.mol2"
ICOLOS_UNITTEST_NIBR_VALUES_KEY = "shape_similarity"

ROCS_SIMILARITY_TEST_DATA = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/reference.sdf"
ROCS_MULTI_SIMILARITY_TEST_DATA = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/reference_multiple_mols.sdf"
ROCS_HIGH_ENERGY_QRY = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/high_energy_conf.sdf"
ROCS_SHAPE_QUERY = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/shape_query.sq"
ROCS_SHAPE_QUERY_2 = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/negative_shape_query.sq"
ROCS_SHAPE_QUERY_3 = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/rocs_bug.sq"
ROCS_SHAPE_QUERY_CFF = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/rocs_phA_rings_only_mod.sq"
ROCS_SHAPE_QUERY_BATCH = '/projects/mai/software/reinvent_data/non_user/test_data/datasets/batch_shape_query.sq'
ROCS_CUSTOM_CFF = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/implicit_MD_mod.cff"
ROCS_NEG_VOL_PROTEIN = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/neg_vol_receptor.pdb"
ROCS_NEG_VOL_LIG = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/neg_vol_ligand.sdf"
ROCS_NEG_VOL_SQ = "/projects/mai/software/reinvent_data/non_user/test_data/datasets/neg_vol_custom_cff.sq"
