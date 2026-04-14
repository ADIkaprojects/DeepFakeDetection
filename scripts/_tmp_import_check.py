import sys
mods = ['torch','torchvision','facenet_pytorch','cv2','sklearn']
for m in mods:
    try:
        __import__(m)
        print(f'IMPORT_OK {m}')
    except Exception as e:
        print(f'IMPORT_FAIL {m}: {e}')
