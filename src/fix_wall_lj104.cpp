/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jonathan Lee (Sandia)
   Modified to add the LJ 10-4 potential by Geordy Jomon (gj82@njit.edu)
   This LJ 10-4 Potential is based on the following paper-
    Siderius, D. W.; Gelb, L. D.
    Extension of the Steele 10-4-3 Potential for Adsorption Calculations in
    Cylindrical, Spherical, and Other Pore Geometries. J. Chem. Phys. 2011,
    135 (8), 084703. https://doi.org/10.1063/1.3626804.
 
    NOTES:
    - This potential is extended for multiple layers.
    - This requires three additonal parameters (See paper for details)-
      - The parameters for the LJ 1043 potential followed by,
      - rho_s: The surface density parameter 
      - n_layers: The number of layers
      - delta_layer: The distance between each layer.
------------------------------------------------------------------------- */

#include "fix_wall_lj104.h"

#include "atom.h"
#include "math_const.h"
#include "math_special.h"

using namespace LAMMPS_NS;
using MathConst::MY_2PI;
using MathSpecial::powint;

/* ---------------------------------------------------------------------- */

FixWallLJ104::FixWallLJ104(LAMMPS *lmp, int narg, char **arg) : FixWall(lmp, narg, arg)
{
  dynamic_group_allow = 1;
}

/* ---------------------------------------------------------------------- */

void FixWallLJ104::precompute(int m)
{
  coeff1[m] = MY_2PI * rho_s[m] * epsilon[m] * sigma[m] * sigma[m];
  coeff2[m] = coeff1[m] * 2.0 / 5.0 * powint(sigma[m], 10);
  coeff3[m] = coeff1[m] * powint(sigma[m], 4);
  
  coeff4[m] = - 10.0 * coeff2[m];
  coeff5[m] = 4.0 * coeff3[m];

  double rinv = 1.0 / cutoff[m];
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r10inv = r4inv * r4inv * r2inv;

  offset[m] = coeff2[m] * r10inv - coeff3[m] * r4inv;
}

/* ---------------------------------------------------------------------- */

void FixWallLJ104::wall_particle(int m, int which, double coord)
{
  double delta, delta_104, rinv, r2inv, r4inv, r5inv, r10inv, r11inv, fwall;
  double vn;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int dim = which / 2;
  int side = which % 2;
  if (side == 0) side = -1;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (side < 0)
        delta = x[i][dim] - coord;
      else
        delta = coord - x[i][dim];
      if (delta <= 0.0) continue;
      if (delta > cutoff[m]) continue;
      fwall = 0;
      for (int layer_index = 0; layer_index < n_layers[m]; layer_index++) {
        delta_104 = delta + layer_index * delta_layer[m];
        rinv = 1.0 / delta_104;
        r2inv = rinv * rinv;
        r4inv = r2inv * r2inv;
        r5inv = r4inv * rinv;
        r10inv = r5inv * r5inv;
        r11inv = r10inv * rinv;

        fwall += - side *
            (coeff4[m] * r11inv + coeff5[m] * r5inv);
        ewall[0] += (coeff2[m] * r10inv - coeff3[m] * r4inv) - offset[m];
      }

      f[i][dim] -= fwall;
      ewall[m + 1] += fwall;
      if (evflag) {
        if (side < 0)
          vn = -fwall * delta;
        else
          vn = fwall * delta;
        v_tally(dim, i, vn);
      }
    }
}
