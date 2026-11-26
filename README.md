# TC-IA-AA — Sistema inteligente de soporte a la decisión de riesgo de impago

Trabajo de Curso de la asignatura **Inteligencia Artificial y Aprendizaje Automático**  
Universidad de las Ciencias Informáticas · Departamento de Inteligencia Computacional

## Objetivo

Desarrollar un sistema que integre las tres áreas del curso en un único problema real:

1. **Módulo 1 — ML:** pipeline de clasificación para predecir impago de tarjeta de crédito.
2. **Módulo 2 — Incertidumbre:** sistema de lógica difusa que traduce la probabilidad del ML en nivel de riesgo y recomendación explicable.
3. **Módulo 3 — Optimización:** algoritmo genético que mejora el pipeline del Módulo 1 (selección de características e hiperparámetros) usando F1 como *fitness*.

La salida final no es solo una etiqueta: es una **decisión con explicación**.

## Decisiones de diseño acordadas

| Elemento | Decisión | Justificación |
|---|---|---|
| Dominio | Finanzas — riesgo de impago | Clase desbalanceada y decisión crítica; conecta directamente con la pregunta de reflexión del PDF |
| Dataset | UCI *Default of Credit Card Clients* (ID 350) | Público, 30 000 registros, 23 variables, licencia CC BY 4.0 |
| Módulo 2 | Lógica difusa (`scikit-fuzzy`) | Convierte una probabilidad en un concepto lingüístico interpretable |
| Módulo 3 | Algoritmo genético (`DEAP`) | Optimiza features/hiperparámetros con `fitness = F1` en validación cruzada |
| Entregable de código | Notebooks + módulos en `src/` | Permite documentar el aprendizaje paso a paso y reutilizar código |

## Dataset y citación

> Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

- Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Variable objetivo: `default` (1 = impago en el siguiente mes, 0 = pago correcto).
- Los datos **no se versionan**: se descargan desde UCI y se guardan en `data/raw/credit_default.csv` mediante `src/data_loader.py`.

## Estructura del repositorio

```text
TC-IA-AA/
├── notebooks/
│   ├── 01_eda.ipynb              # exploración, desbalance, limpieza
│   ├── 02_modelado_ml.ipynb      # Módulo 1: pipeline + 2 modelos + validación cruzada
│   ├── 03_incertidumbre.ipynb    # Módulo 2: sistema difuso
│   ├── 04_optimizacion.ipynb     # Módulo 3: algoritmo genético
│   └── 05_integracion.ipynb      # cadena completa + experimentos + reflexión
├── src/
│   ├── data_loader.py            # descarga/carga y normalización del dataset
│   ├── pipeline.py               # preprocesamiento y modelos (se añadirá)
│   ├── fuzzy_module.py           # razonamiento difuso (se añadirá)
│   └── ga_module.py              # optimización (se añadirá)
├── data/raw/                     # CSV generado, ignorado por Git
├── informe/                      # LaTeX del artículo científico
├── requirements.txt
└── README.md
```

## Puesta en marcha

```powershell
cd "C:\Users\Admin\Desktop\Proyectos\Trabajo Curso IA\TC-IA-AA"
python -m pip install -r requirements.txt --disable-pip-version-check
python -c "import numpy, pandas, sklearn, skfuzzy, deap, ucimlrepo, jupyter; print('Entorno listo')"
python src/data_loader.py
```

Después, abrir los notebooks **en orden numérico**.

## Hoja de ruta

- [x] Leer `orientaciones_TC.pdf` y extraer requisitos
- [x] Elegir dominio, dataset y opciones de los módulos 2 y 3
- [x] Crear `requirements.txt` y estructura base
- [ ] Descargar dataset y hacer EDA (`01_eda.ipynb`)
- [ ] Módulo 1: preprocesamiento, dos modelos y validación cruzada (`02_modelado_ml.ipynb`)
- [ ] Módulo 2: sistema difuso (`03_incertidumbre.ipynb`)
- [ ] Módulo 3: algoritmo genético (`04_optimizacion.ipynb`)
- [ ] Integración y experimentos comparativos (`05_integracion.ipynb`)
- [ ] Informe en LaTeX, README final y diapositivas
- [ ] Hacer público el repositorio y pegar el enlace en la primera página del PDF

## Cómo se logra la integración (criterio del 15 %)

```text
Dataset ──> [M1: Pipeline ML] ──P(impago)──> [M2: Difuso] ──> Decisión + explicación
                ↑                                  │
                └── [M3: GA ajusta features/hiperparámetros] ┘
```

- El **M3 mejora al M1** y se re-entrena.
- El **M2 interpreta al M1** y produce una recomendación.
- La **decisión final** usa ambos, y se compara frente a un baselineAccuracy-only.

## Ética y autoría

El PDF exige declarar explícitamente el uso de asistentes de IA. En el informe se indicará qué herramientas se usaron y con qué propósito (depuración de código, sugerencias de redacción y apoyo didáctico), manteniendo el código y el análisis final como trabajo propio del equipo. El estudiante es responsable al 100 % de la exactitud técnica del contenido entregado.
