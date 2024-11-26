import cv2
import os
import numpy as np
from card_classifier import CardClassifier
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card


class CardDisplay:
    def __init__(self, cards_dir="./card-image"):
        # Inisialisasi path direktori kartu digital
        self.cards_dir = cards_dir
        # Dictionary untuk menyimpan gambar kartu yang sudah di-load
        self.card_images = {}
        # Memuat gambar-gambar kartu dari folder yang ditentukan
        self.load_card_images()

    def load_card_images(self):
        """Memuat semua gambar kartu digital dari direktori"""
        # Cek apakah direktori kartu digital ada
        if not os.path.exists(self.cards_dir):
            print(f"Warning: Directory {self.cards_dir} not found!")
            return

        # Iterasi melalui semua file gambar dengan ekstensi .png
        for filename in os.listdir(self.cards_dir):
            if filename.endswith('.png'):
                # Menghapus ekstensi file untuk mendapatkan nama kartu
                card_name = os.path.splitext(filename)[0]
                # Menggabungkan path dan nama file
                img_path = os.path.join(self.cards_dir, filename)
                # Membaca gambar menggunakan OpenCV
                card_img = cv2.imread(img_path)
                if card_img is not None:
                    # Resize gambar untuk ditampilkan di jendela "Digital Card"
                    card_img = cv2.resize(card_img, (200, 300))
                    # Menyimpan gambar di dictionary dengan nama kartu sebagai key
                    self.card_images[card_name] = card_img

        print(f"Loaded {len(self.card_images)} card images")

    def get_card_image(self, card_class):
        """Mengambil gambar digital berdasarkan kelas kartu yang terdeteksi"""
        try:
            # Memisahkan nama kelas menjadi nilai dan jenis kartu
            value, suit = card_class.split('_', 1)

            # Pemetaan nilai kartu ke singkatan untuk nama file
            value_map = {
                'ace': 'A', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
                'nine': '9', 'ten': '10', 'jack': 'J', 'queen': 'Q',
                'king': 'K'
            }

            # Membuat nama file berdasarkan jenis kartu dan nilai yang dipetakan
            filename = f"{suit}_{value_map[value]}"

            # Mengambil gambar dari dictionary menggunakan nama file
            return self.card_images.get(filename)
        except:
            # Mengembalikan None jika terjadi kesalahan pada format nama kelas
            return None


def main():
    # Inisialisasi classifier dan CardDisplay untuk tampilan kartu digital
    classifier = CardClassifier()
    # Memuat model yang sudah dilatih dari file
    classifier.load_model()
    card_display = CardDisplay()

    # Mengaktifkan kamera untuk mendapatkan video input
    cap = cv2.VideoCapture(2)

    # Membuat jendela tampilan untuk deteksi kartu dan kartu digital
    cv2.namedWindow('Card Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Digital Card', cv2.WINDOW_NORMAL)

    # Menyesuaikan ukuran jendela tampilan
    cv2.resizeWindow('Card Detection', 800, 600)
    cv2.resizeWindow('Digital Card', 200, 300)

    # Variabel untuk menyimpan prediksi terakhir dan threshold confidence
    last_prediction = None
    confidence_threshold = 0.7  # Threshold minimum confidence untuk menampilkan prediksi

    while True:
        # Membaca frame dari kamera
        ret, frame = cap.read()
        if not ret:
            break

        # Mendeteksi kartu pada frame
        card_found, corners, _, _ = detect_card(frame)

        if card_found and corners is not None:
            # Mendapatkan gambar kartu yang ter-warped agar lebih rapi
            _, binary_warped = get_warped_card(frame, corners)

            # Melakukan prediksi menggunakan classifier
            card_class, confidence = classifier.predict(binary_warped)

            # Mengupdate prediksi terakhir jika confidence di atas threshold
            if confidence > confidence_threshold:
                last_prediction = card_class

            # Menampilkan hasil prediksi pada frame utama
            text = f"{card_class} ({confidence:.2%})"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Menampilkan gambar digital dari kartu yang terdeteksi
            if last_prediction:
                digital_card = card_display.get_card_image(last_prediction)
                if digital_card is not None:
                    cv2.imshow('Digital Card', digital_card)

        # Menampilkan frame deteksi
        cv2.imshow('Card Detection', frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan kamera dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
