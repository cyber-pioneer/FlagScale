from abc import ABC, abstractmethod


class BackendBase(ABC):

    @abstractmethod
    def generate_run_script(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_stop_script(self, *args, **kwargs):
        raise NotImplementedError


class MegatronBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class TorchrunBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class VllmBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class SglangBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class LlamallamaCppBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class CustomBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass
