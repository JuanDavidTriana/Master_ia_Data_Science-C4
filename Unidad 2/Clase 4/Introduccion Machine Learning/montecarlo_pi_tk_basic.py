import tkinter as tk
from tkinter import ttk, messagebox
import random
import math

CANVAS_SIZE = 500
RADIUS = CANVAS_SIZE // 2 - 10  # padding from edges
CENTER = CANVAS_SIZE // 2

class MonteCarloPiBasic(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monte Carlo π (básico)")
        self.geometry(f"{CANVAS_SIZE + 260}x{CANVAS_SIZE + 40}")
        self.resizable(False, False)

        self._build_ui()
        self._draw_board()

    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="n")

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="n", padx=(10, 0))

        # Canvas
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        # Controls
        ttk.Label(right, text="Puntos (N):").grid(row=0, column=0, sticky="w")
        self.n_var = tk.StringVar(value="5000")
        self.n_entry = ttk.Entry(right, textvariable=self.n_var, width=12)
        self.n_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(right, text="Semilla (opcional):").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.seed_var = tk.StringVar(value="")
        self.seed_entry = ttk.Entry(right, textvariable=self.seed_var, width=12)
        self.seed_entry.grid(row=1, column=1, sticky="w", pady=(6,0))

        ttk.Label(right, text="Dibujar cada k-ésimo punto:").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.skip_var = tk.StringVar(value="1")
        self.skip_entry = ttk.Entry(right, textvariable=self.skip_var, width=12)
        self.skip_entry.grid(row=2, column=1, sticky="w", pady=(6,0))

        self.run_btn = ttk.Button(right, text="Simular", command=self.run_simulation)
        self.run_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10,0))

        self.clear_btn = ttk.Button(right, text="Limpiar puntos", command=lambda: self.canvas.delete("point"))
        self.clear_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6,0))

        ttk.Separator(right, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        self.pi_label = ttk.Label(right, text="π estimado: -")
        self.pi_label.grid(row=6, column=0, columnspan=2, sticky="w")

        self.err_label = ttk.Label(right, text="Error absoluto: -")
        self.err_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4,0))

        for i in range(2):
            right.grid_columnconfigure(i, weight=1)

    def _draw_board(self):
        # square border
        pad = 10
        self.canvas.create_rectangle(pad, pad, CANVAS_SIZE - pad, CANVAS_SIZE - pad, outline="#333", width=2)
        # inscribed circle
        self.canvas.create_oval(CENTER - RADIUS, CENTER - RADIUS,
                                CENTER + RADIUS, CENTER + RADIUS,
                                outline="#1d4ed8", width=2)

    def run_simulation(self):
        try:
            n = int(self.n_var.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Valor inválido", "Ingrese un entero positivo para N.")
            return

        try:
            k = int(self.skip_var.get())
            if k <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Valor inválido", "'k' debe ser un entero positivo.")
            return

        seed_txt = self.seed_var.get().strip()
        rng = random.Random()
        if seed_txt:
            try:
                rng.seed(int(seed_txt))
            except ValueError:
                rng.seed(seed_txt)  # permitir cadena

        self.canvas.delete("point")

        inside = 0
        r2 = RADIUS * RADIUS

        # generate points in square centered at (CENTER, CENTER)
        for i in range(1, n + 1):
            x = rng.uniform(-RADIUS, RADIUS)
            y = rng.uniform(-RADIUS, RADIUS)
            if x * x + y * y <= r2:
                inside += 1
                color = "#16a34a"  # inside: green
            else:
                color = "#ef4444"  # outside: red

            if i % k == 0:
                cx = CENTER + x
                cy = CENTER - y
                self._draw_point(cx, cy, color)

        pi_est = 4.0 * inside / n
        err = abs(math.pi - pi_est)
        self.pi_label.config(text=f"π estimado: {pi_est:.6f} (con N={n})")
        self.err_label.config(text=f"Error absoluto: {err:.6e}")

    def _draw_point(self, x, y, color):
        s = 2
        self.canvas.create_oval(x - s, y - s, x + s, y + s, fill=color, outline="", tags=("point",))


def main():
    app = MonteCarloPiBasic()
    app.mainloop()


if __name__ == "__main__":
    main()
