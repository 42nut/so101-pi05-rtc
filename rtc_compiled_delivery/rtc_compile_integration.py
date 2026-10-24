"""Opt-in denoise_step compilation for an existing LeRobot PI05 RTC server.

Call install_rtc_compile(policy) AFTER loading policy and initializing RTC,
BEFORE serving inference requests. No robot commands are issued here.

Parity of this checkpoint has NOT passed. The default refuses to enable it.
"""
import logging
import os
import threading
import torch

LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()


def install_rtc_compile(policy, *, enabled=None, allow_unverified=None):
    """Install torch.compile for only PI05 denoise_step; leave RTC guidance eager.

    Returns an informational status dictionary. Does not change model weights.
    - PI05_RTC_COMPILE=1 enables the feature request.
    - PI05_RTC_ALLOW_UNVERIFIED=1 deliberately overrides FAILED offline parity.
    - PI05_RTC_COMPILE_MODE=default selects torch.compile mode.
    Fail closed: do not override failed numerical parity without explicit opt-in.
    """
    if enabled is None:
        enabled = os.getenv('PI05_RTC_COMPILE', '0') == '1'
    if allow_unverified is None:
        allow_unverified = os.getenv('PI05_RTC_ALLOW_UNVERIFIED', '0') == '1'

    if not enabled:
        return {'enabled': False, 'reason': 'disabled; eager path retained'}
    if not allow_unverified:
        LOG.warning('Compiled RTC blocked: checkpoint parity FAILED (9/9); keeping eager.')
        return {'enabled': False, 'reason': 'offline parity failed; explicit override required'}

    model = getattr(policy, 'model', None)
    if model is None or not callable(getattr(model, 'denoise_step', None)):
        raise RuntimeError('Cannot find policy.model.denoise_step; aborting compilation')
    if not callable(getattr(policy, 'predict_action_chunk', None)):
        raise RuntimeError('Expected PI05 policy with predict_action_chunk')
    if hasattr(policy, '_rtc_enabled') and not policy._rtc_enabled():
        raise RuntimeError('RTC is not initialized/enabled; call policy.init_rtc_processor() first')

    with _LOCK:
        if getattr(policy, '_rtc_denoise_compile_installed', False):
            return {'enabled': True, 'reason': 'already installed'}
        eager_fn = model.denoise_step
        mode = os.getenv('PI05_RTC_COMPILE_MODE', 'default')
        compiled_fn = torch.compile(eager_fn, mode=mode, fullgraph=False)
        # Keep eager callable reachable for an explicit STOP -> rollback -> restart.
        policy._rtc_denoise_eager_fn = eager_fn
        policy._rtc_denoise_compiled_fn = compiled_fn
        model.denoise_step = compiled_fn
        policy._rtc_denoise_compile_installed = True
        LOG.warning('UNVERIFIED compiled denoise_step enabled (mode=%s); robot use not validated', mode)
        return {'enabled': True, 'mode': mode, 'parity_verified': False}


def rollback_rtc_compile(policy):
    """Revert to eager; call with inference requests STOPPED, never mid-inference."""
    with _LOCK:
        eager = getattr(policy, '_rtc_denoise_eager_fn', None)
        if eager is None:
            return False
        policy.model.denoise_step = eager
        policy._rtc_denoise_compile_installed = False
        return True
