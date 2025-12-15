import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import random

class DadoRandomState:
    def __init__(self, root):
        self.root = root
        self.root.title("Dado y Random State - Explicación Interactiva")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.historial_lanzamientos_sin_seed = []
        self.historial_lanzamientos_con_seed = []
        self.random_state_value = 42
        self.usa_seed = tk.BooleanVar(value=False)
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Título principal
        titulo = tk.Label(
            self.root, 
            text="🎲 Entendiendo Random State con un Dado",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        titulo.pack(pady=15)
        
        # Marco de explicación
        marco_explicacion = ttk.LabelFrame(
            self.root,
            text="¿Qué es Random State?",
            padding=10
        )
        marco_explicacion.pack(fill="x", padx=15, pady=10)
        
        explicacion = tk.Label(
            marco_explicacion,
            text=("Random State es una semilla (seed) que controla la secuencia de números aleatorios.\n"
                  "Si usas el mismo random_state, obtendrás EXACTAMENTE los mismos resultados.\n"
                  "Esto es esencial para reproducibilidad en Machine Learning."),
            font=("Arial", 10),
            bg="white",
            justify="left",
            wraplength=850
        )
        explicacion.pack(fill="x")
        
        # Marco control
        marco_control = ttk.LabelFrame(
            self.root,
            text="Controles del Dado",
            padding=10
        )
        marco_control.pack(fill="x", padx=15, pady=10)
        
        # Opción de usar random_state
        frame_seed = ttk.Frame(marco_control)
        frame_seed.pack(fill="x", pady=5)
        
        ttk.Checkbutton(
            frame_seed,
            text="Usar Random State (Seed) con valor:",
            variable=self.usa_seed,
            command=self.actualizar_modo
        ).pack(side="left")
        
        ttk.Label(frame_seed, text="Seed:").pack(side="left", padx=(10, 5))
        self.entry_seed = ttk.Entry(frame_seed, width=10)
        self.entry_seed.insert(0, "42")
        self.entry_seed.pack(side="left", padx=5)
        self.entry_seed.config(state="disabled")
        
        # Botones de control
        frame_botones = ttk.Frame(marco_control)
        frame_botones.pack(fill="x", pady=10)
        
        ttk.Button(
            frame_botones,
            text="🎲 Lanzar Dado (Sin Seed)",
            command=self.lanzar_sin_seed
        ).pack(side="left", padx=5)
        
        ttk.Button(
            frame_botones,
            text="🎲 Lanzar Dado (Con Seed)",
            command=self.lanzar_con_seed
        ).pack(side="left", padx=5)
        
        ttk.Button(
            frame_botones,
            text="🔄 Reiniciar",
            command=self.reiniciar
        ).pack(side="left", padx=5)
        
        # Marco de resultados
        marco_resultados = ttk.LabelFrame(
            self.root,
            text="Resultados",
            padding=10
        )
        marco_resultados.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Dos columnas: Sin seed y Con seed
        frame_columnas = ttk.Frame(marco_resultados)
        frame_columnas.pack(fill="both", expand=True)
        
        # Columna izquierda - Sin seed
        frame_izq = ttk.LabelFrame(frame_columnas, text="SIN Random State", padding=10)
        frame_izq.pack(side="left", fill="both", expand=True, padx=5)
        
        self.label_resultado_sin_seed = tk.Label(
            frame_izq,
            text="---",
            font=("Arial", 40, "bold"),
            bg="white",
            fg="#FF6B6B",
            height=3
        )
        self.label_resultado_sin_seed.pack(fill="both", expand=True, pady=10)
        
        ttk.Label(frame_izq, text="Historial:").pack()
        scrollbar_sin = ttk.Scrollbar(frame_izq)
        scrollbar_sin.pack(side="right", fill="y")
        
        self.text_historial_sin_seed = tk.Text(
            frame_izq,
            height=8,
            width=30,
            yscrollcommand=scrollbar_sin.set,
            font=("Courier", 9)
        )
        self.text_historial_sin_seed.pack(fill="both", expand=True)
        scrollbar_sin.config(command=self.text_historial_sin_seed.yview)
        
        # Columna derecha - Con seed
        frame_der = ttk.LabelFrame(frame_columnas, text="CON Random State", padding=10)
        frame_der.pack(side="left", fill="both", expand=True, padx=5)
        
        self.label_resultado_con_seed = tk.Label(
            frame_der,
            text="---",
            font=("Arial", 40, "bold"),
            bg="white",
            fg="#4ECDC4",
            height=3
        )
        self.label_resultado_con_seed.pack(fill="both", expand=True, pady=10)
        
        ttk.Label(frame_der, text="Historial:").pack()
        scrollbar_con = ttk.Scrollbar(frame_der)
        scrollbar_con.pack(side="right", fill="y")
        
        self.text_historial_con_seed = tk.Text(
            frame_der,
            height=8,
            width=30,
            yscrollcommand=scrollbar_con.set,
            font=("Courier", 9)
        )
        self.text_historial_con_seed.pack(fill="both", expand=True)
        scrollbar_con.config(command=self.text_historial_con_seed.yview)
        
        # Marco de conclusión
        marco_conclusion = ttk.LabelFrame(
            self.root,
            text="💡 Conclusion",
            padding=10
        )
        marco_conclusion.pack(fill="x", padx=15, pady=10)
        
        conclusion = tk.Label(
            marco_conclusion,
            text=("✓ SIN Random State: Cada lanzamiento es diferente (impredecible)\n"
                  "✓ CON Random State: Misma semilla = Mismos resultados (reproducible)\n"
                  "✓ En Machine Learning usamos random_state para asegurar que otros puedan reproducir nuestros resultados"),
            font=("Arial", 9),
            bg="white",
            justify="left"
        )
        conclusion.pack(fill="x")
        
    def actualizar_modo(self):
        if self.usa_seed.get():
            self.entry_seed.config(state="normal")
        else:
            self.entry_seed.config(state="disabled")
    
    def lanzar_sin_seed(self):
        try:
            # Sin seed - completamente aleatorio
            resultado = random.randint(1, 6)
            self.historial_lanzamientos_sin_seed.append(resultado)
            
            self.label_resultado_sin_seed.config(text=str(resultado))
            
            # Actualizar historial
            historial_texto = " → ".join(map(str, self.historial_lanzamientos_sin_seed))
            self.text_historial_sin_seed.config(state="normal")
            self.text_historial_sin_seed.delete(1.0, tk.END)
            self.text_historial_sin_seed.insert(1.0, f"Lanzamientos: {len(self.historial_lanzamientos_sin_seed)}\n\n{historial_texto}")
            self.text_historial_sin_seed.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al lanzar: {e}")
    
    def lanzar_con_seed(self):
        try:
            if not self.usa_seed.get():
                messagebox.showwarning("Aviso", "Por favor, activa 'Usar Random State' primero")
                return
            
            seed_value = int(self.entry_seed.get())
            np.random.seed(seed_value)
            resultado = np.random.randint(1, 7)
            self.historial_lanzamientos_con_seed.append(resultado)
            
            self.label_resultado_con_seed.config(text=str(resultado))
            
            # Actualizar historial
            historial_texto = " → ".join(map(str, self.historial_lanzamientos_con_seed))
            self.text_historial_con_seed.config(state="normal")
            self.text_historial_con_seed.delete(1.0, tk.END)
            self.text_historial_con_seed.insert(1.0, f"Seed: {seed_value}\nLanzamientos: {len(self.historial_lanzamientos_con_seed)}\n\n{historial_texto}")
            self.text_historial_con_seed.config(state="disabled")
            
        except ValueError:
            messagebox.showerror("Error", "El valor de seed debe ser un número entero")
    
    def reiniciar(self):
        self.historial_lanzamientos_sin_seed = []
        self.historial_lanzamientos_con_seed = []
        self.label_resultado_sin_seed.config(text="---")
        self.label_resultado_con_seed.config(text="---")
        self.text_historial_sin_seed.config(state="normal")
        self.text_historial_sin_seed.delete(1.0, tk.END)
        self.text_historial_sin_seed.config(state="disabled")
        self.text_historial_con_seed.config(state="normal")
        self.text_historial_con_seed.delete(1.0, tk.END)
        self.text_historial_con_seed.config(state="disabled")

def main():
    root = tk.Tk()
    app = DadoRandomState(root)
    root.mainloop()

if __name__ == "__main__":
    main()
