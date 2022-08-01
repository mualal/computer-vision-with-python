import os

def get_imlist(
    path: str
) -> list:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') and not f.startswith('.')]


if __name__ == '__main__':
    images_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'images_input'
    )
    print(get_imlist(images_path))
    