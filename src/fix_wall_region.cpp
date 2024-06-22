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
   - The dimensions along the axis of the cylinder can vary.

   Notes:
   - The parameters are solid-fluid epsilon and sigma, solid surface-density
     and the cutoff radius.
   - The potential calculation uses the radius of the cylindrical region as
     a parameter.
   - The radius is determined by inspecting the dimensions of the region along
     all three axes. If two dimensions are equal, they define the radial plane,
     and the radius is half of this dimension.
   - If no two dimensions are equal, the region is not cylindrical, and an
     error is returned.

------------------------------------------------------------------------- */

#include "fix_wall_region.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "region.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;
using MathConst::MY_SQRT2;
using MathSpecial::powint;
using MathSpecial::hypergeometric_2F1;

enum { LJ93, LJ126, LJ1043, COLLOID, HARMONIC, MORSE, TJATJOPOULOS };

/* ---------------------------------------------------------------------- */

FixWallRegion::FixWallRegion(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), idregion(nullptr), region(nullptr)
{
  if (narg < 8) utils::missing_cmd_args(FLERR, "fix wall/region", error);

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
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", arg[3]);
  idregion = utils::strdup(arg[3]);

  if (strcmp(arg[4], "lj93") == 0)
    style = LJ93;
  else if (strcmp(arg[4], "lj126") == 0)
    style = LJ126;
  else if (strcmp(arg[4], "lj1043") == 0)
    style = LJ1043;
  else if (strcmp(arg[4], "colloid") == 0)
    style = COLLOID;
  else if (strcmp(arg[4], "harmonic") == 0)
    style = HARMONIC;
  else if (strcmp(arg[4], "morse") == 0)
    style = MORSE;
  else if (strcmp(arg[4], "tjatjopoulos") == 0)
    style = TJATJOPOULOS;
  else
    error->all(FLERR, "Unknown fix wall/region style {}", arg[4]);

  if (style != COLLOID) dynamic_group_allow = 1;

  if (style == MORSE) {
    if (narg != 9) error->all(FLERR, "Illegal fix wall/region morse command");

    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    alpha = utils::numeric(FLERR, arg[6], false, lmp);
    sigma = utils::numeric(FLERR, arg[7], false, lmp);
    cutoff = utils::numeric(FLERR, arg[8], false, lmp);

  } else if (style == TJATJOPOULOS) {
    if (narg != 9) error->all(FLERR, "Illegal fix wall/region tjatjopoulos command, incorrect number of arguments.");
 
    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    sigma = utils::numeric(FLERR, arg[6], false, lmp);
    rho_A = utils::numeric(FLERR, arg[7], false, lmp);
    cutoff = utils::numeric(FLERR, arg[8], false, lmp);

  } else {
    if (narg != 8) error->all(FLERR, "Illegal fix wall/region command");

    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    sigma = utils::numeric(FLERR, arg[6], false, lmp);
    cutoff = utils::numeric(FLERR, arg[7], false, lmp);
  }

  if (cutoff <= 0.0) error->all(FLERR, "Fix wall/region cutoff <= 0.0");

  eflag = 0;
  ewall[0] = ewall[1] = ewall[2] = ewall[3] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixWallRegion::~FixWallRegion()
{
  delete[] idregion;
}

/* ---------------------------------------------------------------------- */

int FixWallRegion::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallRegion::init()
{
  // set index and check validity of region

  region = domain->get_region_by_id(idregion);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", idregion);

  // error checks for style COLLOID
  // ensure all particles in group are extended particles

  if (style == COLLOID) {
    if (!atom->radius_flag) error->all(FLERR, "Fix wall/region colloid requires atom attribute radius");

    double *radius = atom->radius;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (radius[i] == 0.0) flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
    if (flagall) error->all(FLERR, "Fix wall/region colloid requires only extended particles");
  }

  // setup coefficients for each style

  if (style == LJ93) {
    coeff1 = 6.0 / 5.0 * epsilon * powint(sigma, 9);
    coeff2 = 3.0 * epsilon * powint(sigma, 3);
    coeff3 = 2.0 / 15.0 * epsilon * powint(sigma, 9);
    coeff4 = epsilon * powint(sigma, 3);
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv;
  } else if (style == LJ126) {
    coeff1 = 48.0 * epsilon * powint(sigma, 12);
    coeff2 = 24.0 * epsilon * powint(sigma, 6);
    coeff3 = 4.0 * epsilon * powint(sigma, 12);
    coeff4 = 4.0 * epsilon * powint(sigma, 6);
    double r2inv = 1.0 / (cutoff * cutoff);
    double r6inv = r2inv * r2inv * r2inv;
    offset = r6inv * (coeff3 * r6inv - coeff4);
  } else if (style == LJ1043) {
    coeff1 = MY_2PI * 2.0 / 5.0 * epsilon * powint(sigma, 10);
    coeff2 = MY_2PI * epsilon * powint(sigma, 4);
    coeff3 = MY_2PI * MY_SQRT2 / 3.0 * epsilon * powint(sigma, 3);
    coeff4 = 0.61 / MY_SQRT2 * sigma;
    coeff5 = coeff1 * 10.0;
    coeff6 = coeff2 * 4.0;
    coeff7 = coeff3 * 3.0;
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff1 * r4inv * r4inv * r2inv - coeff2 * r4inv - coeff3 * powint(cutoff + coeff4, -3);
  } else if (style == MORSE) {
    coeff1 = 2 * epsilon * alpha;
    double alpha_dr = -alpha * (cutoff - sigma);
    offset = epsilon * (exp(2.0 * alpha_dr) - 2.0 * exp(alpha_dr));
  } else if (style == COLLOID) {
    coeff1 = -4.0 / 315.0 * epsilon * powint(sigma, 6);
    coeff2 = -2.0 / 3.0 * epsilon;
    coeff3 = epsilon * powint(sigma, 6) / 7560.0;
    coeff4 = epsilon / 6.0;
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv;
  } else if (style == TJATJOPOULOS) {
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
    double sigma_R = sigma / R;
    tjat_coeff = MY_2PI * rho_A * sigma * sigma * epsilon;
    psi6_coeff = psi6_gc * powint(sigma_R, 10); // 10 and 4 from 2n-2
    psi3_coeff = psi3_gc * powint(sigma_R, 4);
    psi6_der1 = 40.5 * powint(R,18);
    psi6_der2 = 0.493827 * R * R;
    psi3_der1 = 4.5 * powint(R,6);
    psi3_der2 = 8 * powint(R,8);
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallRegion::setup(int vflag)
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

void FixWallRegion::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallRegion::post_force(int vflag)
{
  int i, m, n;
  double rinv, fx, fy, fz, tooclose;
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
  // in COLLOID case, r <= radius is an error
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
      if (style == COLLOID)
        tooclose = radius[i];
      else
        tooclose = 0.0;

      n = region->surface(x[i][0], x[i][1], x[i][2], cutoff);

      for (m = 0; m < n; m++) {
        if (region->contact[m].r <= tooclose) {
          onflag = 1;
          continue;
        } else
          rinv = 1.0 / region->contact[m].r;

        switch (style) {
          case TJATJOPOULOS:
            tjatjopoulos(region->contact[m].r);
            break;
          case LJ93:
            lj93(region->contact[m].r);
            break;
          case LJ1043:
            lj1043(region->contact[m].r);
            break;
          case MORSE:
            morse(region->contact[m].r);
            break;
          case COLLOID:
            colloid(region->contact[m].r, radius[i]);
            break;
          case HARMONIC:
            harmonic(region->contact[m].r);
            break;
          default:
            lj126(region->contact[m].r);
        }

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

void FixWallRegion::post_force_respa(int vflag, int ilevel, int /* iloop */)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallRegion::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallRegion::compute_scalar()
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

double FixWallRegion::compute_vector(int n)
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall, ewall_all, 4, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return ewall_all[n + 1];
}

/* ----------------------------------------------------------------------
   LJ 9/3 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::lj93(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r10inv = r4inv * r4inv * r2inv;
  fwall = coeff1 * r10inv - coeff2 * r4inv;
  eng = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv - offset;
}

/* ----------------------------------------------------------------------
   LJ 12/6 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::lj126(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r6inv = r2inv * r2inv * r2inv;
  fwall = r6inv * (coeff1 * r6inv - coeff2) * rinv;
  eng = r6inv * (coeff3 * r6inv - coeff4) - offset;
}

/* ----------------------------------------------------------------------
   LJ 10/4/3 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::lj1043(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r10inv = r4inv * r4inv * r2inv;
  fwall = coeff5 * r10inv * rinv - coeff6 * r4inv * rinv - coeff7 * powint(r + coeff4, -4);
  eng = coeff1 * r10inv - coeff2 * r4inv - coeff3 * powint(r + coeff4, -3) - offset;
}

/* ----------------------------------------------------------------------
   Morse interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::morse(double r)
{
  double dr = r - sigma;
  double dexp = exp(-alpha * dr);
  fwall = coeff1 * (dexp * dexp - dexp);
  eng = epsilon * (dexp * dexp - 2.0 * dexp) - offset;
}

/* ----------------------------------------------------------------------
   colloid interaction for finite-size particle of rad with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::colloid(double r, double rad)
{
  double new_coeff2 = coeff2 * rad * rad * rad;
  double diam = 2.0 * rad;

  double rad2 = rad * rad;
  double rad4 = rad2 * rad2;
  double rad8 = rad4 * rad4;
  double delta2 = rad2 - r * r;
  double rinv = 1.0 / delta2;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r8inv = r4inv * r4inv;
  fwall = coeff1 *
          (rad8 * rad + 27.0 * rad4 * rad2 * rad * r * r + 63.0 * rad4 * rad * powint(r, 4) +
           21.0 * rad2 * rad * powint(r, 6)) *
          r8inv -
      new_coeff2 * r2inv;

  double r2 = 0.5 * diam - r;
  double rinv2 = 1.0 / r2;
  double r2inv2 = rinv2 * rinv2;
  double r4inv2 = r2inv2 * r2inv2;
  double r3 = r + 0.5 * diam;
  double rinv3 = 1.0 / r3;
  double r2inv3 = rinv3 * rinv3;
  double r4inv3 = r2inv3 * r2inv3;
  eng = coeff3 *
          ((-3.5 * diam + r) * r4inv2 * r2inv2 * rinv2 +
           (3.5 * diam + r) * r4inv3 * r2inv3 * rinv3) -
      coeff4 * ((-diam * r + r2 * r3 * (log(-r2) - log(r3))) * (-rinv2) * rinv3) - offset;
}

/* ----------------------------------------------------------------------
   harmonic interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallRegion::harmonic(double r)
{
  double dr = cutoff - r;
  fwall = 2.0 * epsilon * dr;
  eng = epsilon * dr * dr;
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

void FixWallRegion::tjatjopoulos(double r)
{
  double dr = R - r;
  double dr_R = dr / R;
  double dr2_R2 = dr_R * dr_R; 
  double omdr2_R2 = 1 - dr2_R2;

  double psi6_2F1 = hypergeometric_2F1(-4.5,-4.5,1,dr2_R2);
  double psi3_2F1 = hypergeometric_2F1(-1.5, -1.5, 1, dr2_R2);
  double psi6 = psi6_coeff * powint(omdr2_R2, -10) * psi6_2F1;
  double psi3 = psi3_coeff * powint(omdr2_R2, -4) * psi3_2F1;

  eng = tjat_coeff * (psi6 - psi3);

  double dr2m_R2 = dr * dr - R * R;
  double psi6_der = psi6_coeff * (((dr * psi6_der1) *
    ((dr2m_R2 * hypergeometric_2F1(-3.5,-3.5,2,dr2_R2)) - (psi6_der2 * psi6_2F1)))
    / powint(dr2m_R2, 11));
  double psi3_der = psi3_coeff * (((dr * psi3_der1 * hypergeometric_2F1(-0.5, -0.5, 2, dr2_R2))
    - (dr * psi3_der2 * psi3_2F1))
    / powint(dr2m_R2, 5));

  fwall = tjat_coeff * (psi6_der - psi3_der);
}
