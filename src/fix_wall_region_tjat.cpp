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
   Modified by: Geordy Jomon (gj82@njit.edu)

   Description:
   This modification introduces the Tjatjopoulos potential for cylindrical
   regions into the simulation. The equations are derived from 'Extension
   of the Steele 10-4-3 potential' by Siderius and Gelb (2011), specifically
   Equation 5.

   Restrictions:
   - This potential is applicable only to cylindrical regions with a constant
     radius.
   - This implementation was designed with the axis of the cylinder being
     perodic. This can be achived by setting the height of the region as INF
     in the lower and upper bound.

   Notes:
   - The parameters are solid-fluid epsilon and sigma, solid surface-density
     and the cutoff radius.
   - The dimensions along the axis of the cylinder can vary, i.e there can be
     volume fluctuations in that axis.
   - The potential calculation uses the radius of the cylindrical region as
     a parameter.
   - The radius is determined by inspecting the dimensions of the region along
     all three axes. If two dimensions are equal, they define the radial plane,
     and the radius is half of this dimension.
   - If no two dimensions are equal, the region is not cylindrical, and an
     error is returned.
   - This is still in testing phase and while there is agreement in one
     system that was tested, rigrous testing is required.

------------------------------------------------------------------------- */

#include "fix_wall_region_tjat.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "math_extra.h"
#include "region.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;
using MathConst::MY_PIS;
using MathSpecial::powint;
using MathExtra::hypergeometric_2F1;

/* ---------------------------------------------------------------------- */

FixWallRegionTjat::FixWallRegionTjat(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), idregion(nullptr), region(nullptr)
{
  if (narg != 8) error->all(FLERR, "Illegal fix wall/region/tjatjopoulos command");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  // parse args

  region = domain->get_region_by_id(arg[3]);
  if (!region) error->all(FLERR, "Region {} for fix wall/region/tjatjopoulos does not exist", arg[3]);
  idregion = utils::strdup(arg[3]);
 
  epsilon = utils::numeric(FLERR, arg[4], false, lmp);
  sigma = utils::numeric(FLERR, arg[5], false, lmp);
  rho_A = utils::numeric(FLERR, arg[6], false, lmp);
  cutoff = utils::numeric(FLERR, arg[7], false, lmp);

  if (cutoff <= 0.0) error->all(FLERR, "Fix wall/region cutoff <= 0.0");

  eflag = 0;
  ewall[0] = ewall[1] = ewall[2] = ewall[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixWallRegionTjat::~FixWallRegionTjat()
{
  delete[] idregion;
}

/* ---------------------------------------------------------------------- */

int FixWallRegionTjat::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::init()
{
  // set index and check validity of region

  region = domain->get_region_by_id(idregion);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", idregion);
  if (region->varshape) error->all(FLERR, "fix wall/region tjatjopoulos: Region {} must be static", idregion);

  // Radius of cylinder

  double xprd_half = (region->extent_xhi - region->extent_xlo) / 2;
  double yprd_half = (region->extent_yhi - region->extent_ylo) / 2;
  double zprd_half = (region->extent_zhi - region->extent_zlo) / 2;
  if (yprd_half == zprd_half || yprd_half == xprd_half) {
    R = yprd_half;
  } else if (zprd_half == xprd_half) {
    R = zprd_half;
  } else {
    error->all(FLERR, "fix wall/region Tjatjopoulos:  {} should have uniform dimensions in atleast one pane x:{}, y:{}, z:{}", idregion, xprd_half, yprd_half, zprd_half);
  }

  // setup coefficients

  const double psi6_gc = 4 * MY_PIS * tgamma(5.5) / tgamma(6);
  const double psi3_gc = 4 * MY_PIS * tgamma(2.5) / tgamma(3);
  double sigma_R = sigma / R;

  R2 = R * R;
  tjat_coeff = MY_2PI * rho_A * sigma * sigma * epsilon;
  psi6_coeff = psi6_gc * powint(sigma_R, 10); // 10 and 4 from 2n-2
  psi3_coeff = psi3_gc * powint(sigma_R, 4);
  psi6_der1 = 40.5 * powint(R,18);
  psi6_der2 = 0.493827 * R * R;
  psi3_der1 = 4.5 * powint(R,6);
  psi3_der2 = 8 * powint(R,8);

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}


/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    respa->copy_f_flevel(ilevel_respa);
  } else {
    post_force(vflag);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::post_force(int vflag)
{
  int i, m, n;
  double rinv, fx, fy, fz;
  double delx, dely, delz, v[6];

  double **x = atom->x;
  double **f = atom->f;
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  region->prematch();

  int onflag = 0;

  // virial setup

  v_init(vflag);

  // region->match() ensures particle is in region or on surface, else error
  // if returned contact dist r = 0, is on surface, also an error
  // initilize ewall after region->prematch(),
  //   so a dynamic region can access last timestep values

  eflag = 0;
  ewall[0] = ewall[1] = ewall[2] = ewall[3] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (!region->match(x[i][0], x[i][1], x[i][2])) {
        onflag = 1;
        continue;
      }

      n = region->surface(x[i][0], x[i][1], x[i][2], cutoff);

      for (m = 0; m < n; m++) {
        if (region->contact[m].r <= 0.0) {
          onflag = 1;
          continue;
        } else
          rinv = 1.0 / region->contact[m].r;

        tjatjopoulos(region->contact[m].r);

        delx = region->contact[m].delx;
        dely = region->contact[m].dely;
        delz = region->contact[m].delz;
        fx = fwall * delx * rinv;
        fy = fwall * dely * rinv;
        fz = fwall * delz * rinv;
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        ewall[1] -= fx;
        ewall[2] -= fy;
        ewall[3] -= fz;
        ewall[0] += eng;
        if (evflag) {
          v[0] = fx * delx;
          v[1] = fy * dely;
          v[2] = fz * delz;
          v[3] = fx * dely;
          v[4] = fx * delz;
          v[5] = fy * delz;
          v_tally(i, v);
        }
      }
    }

  if (onflag) error->one(FLERR, "Particle outside surface of region used in fix wall/region");
}

/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::post_force_respa(int vflag, int ilevel, int /* iloop */)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallRegionTjat::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallRegionTjat::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall, ewall_all, 4, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return ewall_all[0];
}

/* ----------------------------------------------------------------------
   components of force on wall
------------------------------------------------------------------------- */

double FixWallRegionTjat::compute_vector(int n)
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall, ewall_all, 4, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return ewall_all[n + 1];
}

/* ----------------------------------------------------------------------
   Tjatjopoulos cylindrical interaction for particle with wall.
   compute eng and fwall = magnitude of wall force. Equations are from
   'Extension of the Steele 10-4-3 potential' Siderius & Gelb 2011 Eq.(5)
   hypergeometric_2F1 is the gauss hypergeometric function for restricted
   domain: c > 0 and abs(z) < 1. This potential considers the distance of
   the particle from the center of the cylinder dr = R - r where r is the
   distance from the surface, R is the radius of the cylindrical region
------------------------------------------------------------------------- */

void FixWallRegionTjat::tjatjopoulos(double r)
{
  double dr = R - r;
  double dr2 = dr * dr;
  double dr2_R2 = dr2 / R2;
  double ov_omdr2_R2 = 1.0 / (1.0 - dr2_R2);
  double ov_omdr2_R2_2 = ov_omdr2_R2 * ov_omdr2_R2;
  double ov_omdr2_R2_4 = ov_omdr2_R2_2 * ov_omdr2_R2_2;
  double ov_omdr2_R2_10 = ov_omdr2_R2_4 * ov_omdr2_R2_4 * ov_omdr2_R2_2;

  double psi6_2F1 = hypergeometric_2F1(-4.5, -4.5, 1.0, dr2_R2);
  double psi3_2F1 = hypergeometric_2F1(-1.5, -1.5, 1.0, dr2_R2);
  double psi6 = psi6_coeff * ov_omdr2_R2_10 * psi6_2F1;
  double psi3 = psi3_coeff * ov_omdr2_R2_4 * psi3_2F1;

  eng = tjat_coeff * (psi6 - psi3);

  double dr2m_R2 = dr2 - R2;
  double dr2m_R2_2 = dr2m_R2 * dr2m_R2;
  double dr2m_R2_5 = dr2m_R2_2 * dr2m_R2_2 * dr2m_R2;
  double dr2m_R2_11 = dr2m_R2_5 * dr2m_R2_5 * dr2m_R2;

  double psi6_der = psi6_coeff * (((dr * psi6_der1) *
    ((dr2m_R2 * hypergeometric_2F1(-3.5, -3.5, 2.0, dr2_R2)) - (psi6_der2 * psi6_2F1)))
    / dr2m_R2_11);
  double psi3_der = psi3_coeff * (((dr * psi3_der1 * hypergeometric_2F1(-0.5, -0.5, 2.0, dr2_R2))
    - (dr * psi3_der2 * psi3_2F1))
    / dr2m_R2_5);

  fwall = tjat_coeff * (psi6_der - psi3_der);
}
