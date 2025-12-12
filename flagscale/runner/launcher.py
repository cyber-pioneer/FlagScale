from abc import ABC, abstractmethod


class LauncherBase(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def stop(self, *args, **kwargs):
        raise NotImplementedError


class SshLauncher(LauncherBase):
    def run(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


class CloudLauncher(LauncherBase):
    def run(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass
