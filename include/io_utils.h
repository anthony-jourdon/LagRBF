/*@
====================================================================================================
  Copyright (c) 2026
      Anthony Jourdon jourdon@geoazur.unice.fr
      Université Côte d'Azur
      Observatoire de la Côte d'Azur, CNRS, IRD, Géoazur
      250 Rue Albert Einstein
      06560 Valbonne
      France
====================================================================================================

  project: LagRBF
  file:    io_utils.h

  This file is part of LagRBF software.
  LagRBF is free software: you can redistribute it and/or modify it under the terms of the GNU
  General Public License as published by the Free Software Foundation, either version 3 of the
  License, or any later version.

  LagRBF is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with LagRBF.
  If not, see <https://www.gnu.org/licenses/>.
@*/

#ifndef __LagRBF_io_utils_h
#define __LagRBF_io_utils_h

#include <petsc.h>

PetscErrorCode CreateDirectory(const char dirname[]);
PetscErrorCode TestDirectory(const char dirname[], char mode, PetscBool *_exists);
PetscErrorCode TestFile(const char filename[], char mode, PetscBool *_exists);
PetscErrorCode ParaviewPVDOpen(const char pvdfilename[]);
PetscErrorCode ParaviewPVDAppend(const char pvdfilename[], double time, const char datafile[], const char directory_name[]);
PetscErrorCode ParaviewPVDOpenAppend(PetscBool not_first, PetscInt step, const char pvdfilename[], double time, const char datafile[], const char directory_name[]);
PetscErrorCode VecLoadFromFile(Vec v, const char *name);
PetscErrorCode ISLoadFromFile(IS is, const char *name);
PetscErrorCode GetTimeStepsFromOptions(PetscInt *nsteps, PetscInt **steps, PetscReal **time_steps);
PetscErrorCode RestoreTimeSteps(PetscInt **steps, PetscReal **time_steps);
PetscErrorCode BuildFilenameFromTemplate(const char template_name[], const char directory[], const char prefix[], const char field[], const char separator[], PetscInt step, char out[]);
PetscErrorCode GetFilelistFromOptions(const char *prefix, PetscInt step, PetscInt *nfiles, char ***filenames, char **coorfname);
PetscErrorCode RestoreFilelist(PetscInt nfiles, char ***filenames, char **coorfname);
#endif