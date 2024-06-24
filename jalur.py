import os

file_path = "audios/en/Srikaya.mp3"  # Ganti dengan jalur file yang ingin Anda periksa

if os.path.exists(file_path):
    print("File ada.")
else:
    print("File tidak ada atau jalur file salah.")
