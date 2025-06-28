import importlib
import sys
import types


def test_helper_functions_exist(monkeypatch):
    airsim_stub = types.SimpleNamespace(ImageRequest=object, ImageType=object)
    monkeypatch.setitem(sys.modules, 'airsim', airsim_stub)
    nl = importlib.import_module('uav.nav_loop')
    importlib.reload(nl)
    for name in (
        'setup_environment',
        'start_perception_thread',
        'navigation_loop',
        'process_perception_data',
        'apply_navigation_decision',
        'write_frame_output',
        'handle_reset',
        'cleanup',
    ):
        assert hasattr(nl, name)
