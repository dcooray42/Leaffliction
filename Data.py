class FolderData:
    def __init__(self, path):
        self.__path = path
        self.__count = 0
        self.__sub_dir = []

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"path : {self.__path}, count : {self.__count}"

    def get_path(self) -> str:
        return self.__path

    def get_count(self) -> int:
        return self.__count

    def get_sub_dir(self) -> list:
        return self.__sub_dir

    def add_count(self, count):
        self.__count += count

    def add_sub_dir(self, sub_dir):
        self.__sub_dir.append(sub_dir)
