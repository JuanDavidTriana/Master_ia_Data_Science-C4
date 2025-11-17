
"""Funciones de análisis de productos.

Un producto es un diccionario con claves: {"nombre", "precio", "cantidad"}.
"""
from typing import List, Dict, Optional

Producto = Dict[str, float]

def total_vendido(productos: List[Producto]) -> float:
    """Devuelve el total vendido = sum(precio * cantidad)."""
    total = 0.0
    for p in productos:
        precio = p.get("precio")
        cantidad = p.get("cantidad")
        if precio is None or cantidad is None:
            raise ValueError("Producto sin precio o cantidad")
        total += float(precio) * float(cantidad)
    return round(total, 2)

def promedio_precios(productos: List[Producto]) -> float:
    """Devuelve el precio promedio de los productos (independiente de la cantidad)."""
    precios = [float(p["precio"]) for p in productos if "precio" in p]
    if not precios:
        raise ValueError("Lista de productos sin precios")
    return round(sum(precios) / len(precios), 2)

def filtrar_por_precio(productos: List[Producto], minimo: float, maximo: Optional[float] = None) -> List[Producto]:
    """Filtra productos por rango de precio [minimo, maximo]. Si maximo es None, solo aplica el mínimo."""
    res = []
    for p in productos:
        precio = float(p.get("precio", 0))
        if precio >= minimo and (maximo is None or precio <= maximo):
            res.append(p)
    return res

def reporte(productos: List[Producto]) -> Dict[str, float]:
    """Devuelve un reporte con totales y promedios."""
    return {
        "total_vendido": total_vendido(productos),
        "promedio_precios": promedio_precios(productos),
        "num_items": len(productos),
    }
