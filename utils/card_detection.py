import cv2
import numpy as np


def detect_card(frame):
    # Mengonversi gambar asli (BGR) menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Menggunakan Gaussian Blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mendeteksi tepi dalam gambar menggunakan algoritma Canny
    edged = cv2.Canny(blurred, 50, 200)

    # Melakukan dilasi pada hasil deteksi tepi untuk mempertebal garis tepi
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Mengambil kontur dari gambar hasil dilasi
    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menyiapkan gambar preview dari kontur yang terdeteksi
    contour_preview = np.zeros_like(edged)
    cv2.drawContours(contour_preview, contours, -1, (255, 255, 255), 2)

    # Mengecek apakah ada kontur yang terdeteksi
    if len(contours) > 0:
        # Mengambil kontur terbesar, yang diasumsikan sebagai kartu
        card_contour = max(contours, key=cv2.contourArea)

        # Memastikan area kontur cukup besar (dalam piksel)
        if cv2.contourArea(card_contour) > 5000:
            # Menghitung panjang keliling kontur
            peri = cv2.arcLength(card_contour, True)
            # Mendapatkan bentuk kontur yang disederhanakan dengan 4 titik sudut
            approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)

            # Jika kontur memiliki 4 titik sudut, kemungkinan itu adalah kartu
            if len(approx) == 4:
                # Menggambar kontur yang terdeteksi pada frame asli
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                # Menyimpan titik sudut kontur dalam bentuk array
                corners = approx.reshape(4, 2)

                # Menandai setiap titik sudut dengan lingkaran merah
                for corner in corners:
                    x, y = corner.astype(int)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Mengembalikan True, posisi sudut-sudut kontur, preview kontur, dan gambar tepi
                return True, corners, contour_preview, edged

    # Jika tidak ada kontur yang sesuai ditemukan
    return False, None, contour_preview, edged
