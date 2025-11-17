
from analisis import total_vendido, promedio_precios, filtrar_por_precio, reporte

def main():
    productos = [
        {"nombre": "A", "precio": 10.0, "cantidad": 3},
        {"nombre": "B", "precio": 25.0, "cantidad": 2},
        {"nombre": "C", "precio": 15.0, "cantidad": 5},
    ]
    print("Total vendido:", total_vendido(productos))
    print("Promedio de precios:", promedio_precios(productos))
    print("Filtrados >= 15:", filtrar_por_precio(productos, minimo=15.0))
    print("Reporte:", reporte(productos))

if __name__ == "__main__":
    main()
