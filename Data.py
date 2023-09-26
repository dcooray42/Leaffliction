class FolderData:
    def __init__(self, path, count):
        self.__path = path
        self.__count = count

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"path : {self.__path}, count : {self.__count}"
