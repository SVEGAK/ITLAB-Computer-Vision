import argparse
import cv2
import sys
import os
#Примеры команд (для текущей директории)
# =============================================================================
# !python main.py --help
# !python main.py -i cat.jpg -f resize -w 800 --height 600
# !python main.py -i cat.jpg -f neon --B 0.9 --G 0.2 --R 0.6 
# !python main.py -i cat.jpg -f grayscale
# !python main.py -i cat.jpg -f oldPhoto -e 0.5
# !python main.py -i cat.jpg -f fadeColor -e 0.8
# !python main.py -i cat.jpg -f antique
# !python main.py -i cat.jpg -f mask
# !python main.py -i cat.jpg -f negativeFilm
# =============================================================================
from filters import (
    resize_img, convert_to_grayscale, fade_color, 
    old_photography, antique_filter, mask, 
    negative_film, neon_effect
)

def main():
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Приложение для применения фильтров к изображениям.")
    
    # Обязательные аргументы
    parser.add_argument("-i", "--image", required=True, help="Путь к исходному изображению")
    parser.add_argument("-f", "--filter", required=True, 
                        choices=["resize", "grayscale", "oldPhoto", "fadeColor", "antique", "mask", "negativeFilm", "neon"], 
                        help="Какой фильтр применить")
    
    # Необязательные аргументы (общие)
    parser.add_argument("-o", "--output", default="result.jpg", help="Путь для сохранения результата (по умолчанию: result.jpg)")
    
    # Специфичные аргументы для фильтров
    parser.add_argument("-e", "--extent", type=float, default=0.8, help="Параметр extent для фильтров oldPhoto и fadeColor (по умолчанию: 0.8)")
    parser.add_argument("-w", "--width", type=int, default=500, help="Ширина для фильтра resize (по умолчанию: 500)")
    parser.add_argument("--height", type=int, default=500, help="Высота для фильтра resize (по умолчанию: 500)")
    
    # Коэффициенты для неона (по умолчанию берем те, что ты использовал ранее)
    parser.add_argument("--B", type=float, default=1.0, help="Синий коэффициент для неона (по умолчанию: 1.0)")
    parser.add_argument("--G", type=float, default=0.1, help="Зеленый коэффициент для неона (по умолчанию: 0.1)")
    parser.add_argument("--R", type=float, default=0.4, help="Красный коэффициент для неона (по умолчанию: 0.4)")

    # 2. Парсим аргументы
    args = parser.parse_args()

    # 3. Проверяем, существует ли файл
    if not os.path.exists(args.image):
        print(f"Ошибка: Файл '{args.image}' не найден.")
        sys.exit(1)

    # 4. Загружаем изображение
    img = cv2.imread(args.image)
    if img is None:
        print(f"Ошибка: Не удалось прочитать изображение '{args.image}'. Возможно, это не картинка.")
        sys.exit(1)

    print(f"Обработка изображения... Применяем фильтр: {args.filter}")

    # 5. Применяем нужный фильтр
    if args.filter == "resize":
        res = resize_img(img, args.width, args.height)
        
    elif args.filter == "grayscale":
        res = convert_to_grayscale(img)
        
    elif args.filter == "oldPhoto":
        res = old_photography(img, extent=args.extent)
        
    elif args.filter == "fadeColor":
        res = fade_color(img, extent=args.extent)
        
    elif args.filter == "antique":
        res = antique_filter(img)
        
    elif args.filter == "mask":
        res = mask(img)
        
    elif args.filter == "negativeFilm":
        res = negative_film(img)
        
    elif args.filter == "neon":
        res = neon_effect(img, args.B, args.G, args.R)
        
    else:
        print("Неизвестный фильтр.")
        sys.exit(1)

    # 6. Сохраняем результат
    cv2.imwrite(args.output, res)
    print(f"Готово! Результат сохранен в файл: {args.output}")

    # 7. Показываем результат на экране
    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()