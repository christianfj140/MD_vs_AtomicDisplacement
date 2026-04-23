# Pseudopotentials

Guarda aquí los pseudopotenciales que use SIESTA para este proyecto.

## Recomendación para este repositorio (fase actual)

- Carpeta recomendada: `MD/dataset/pseudopotentials/`
- Archivos típicos: `H.psf`, `O.psf` (o equivalentes según los elementos del sistema).

## Nota importante sobre los scripts actuales

En el flujo hardcodeado actual, `siesta` se ejecuta dentro de `MD/dataset/`.
Por eso, **SIESTA buscará los pseudopotenciales en el directorio de trabajo**
(si no se define otra ruta en `RUN.fdf`).

En esta fase puedes:

1. Mantener una copia "fuente" en `MD/dataset/pseudopotentials/`.
2. Copiar (o enlazar) los `.psf` necesarios a `MD/dataset/` antes de ejecutar.

Esto mantiene ordenado el repo y conserva compatibilidad con tu pipeline actual.
