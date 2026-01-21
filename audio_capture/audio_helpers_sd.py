import numpy as np
import sounddevice as sd

LOOPBACK_KEYWORDS = [
    'monitor', 'loopback', # Linux only
]

class AudioClass:
    def __init__(self, *args, **kw):
        pass

    def get_available_devices(self):
        devices = {}
        try:
            devices = sd.query_devices()
        except Exception as e:
            raise(f"error getting audio devices: {e}")
        return devices

    def get_default_loopback(self) -> dict[str:any]:
        """Get default loopback device"""
        try:
            return 


    def set_asio_out(self):
        try:
            asio_in = sd.AsioSettings(channel_selectors=[8])
            asio_out = sd.AsioSettings(channel_selectors=[12, 13])
            sd.default.extra_settings = asio_in, asio_out
        except Exception as e:
            raise(f"Error setting asio drivers: {e}")
        return asio_in
