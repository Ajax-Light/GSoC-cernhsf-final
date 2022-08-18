#include "eicd/CalorimeterHit.h"
#include "eicd/CalorimeterHitCollection.h"
#include "eicd/ReconstructedParticleCollection.h"
#include <string>
#include <vector>

#define debug() std::cout
#define info() std::cout
#define verbose() std::cout
#define error() std::cerr
#define endmsg std::endl

namespace my_units {
// Make units similar to Gaudi::Units
//
// Length [L]
//
constexpr double millimeter  = 1.;
constexpr double millimeter2 = millimeter * millimeter;
constexpr double millimeter3 = millimeter * millimeter * millimeter;

constexpr double centimeter  = 10. * millimeter;
constexpr double centimeter2 = centimeter * centimeter;
constexpr double centimeter3 = centimeter * centimeter * centimeter;

constexpr double meter  = 1000. * millimeter;
constexpr double meter2 = meter * meter;
constexpr double meter3 = meter * meter * meter;

constexpr double kilometer  = 1000. * meter;
constexpr double kilometer2 = kilometer * kilometer;
constexpr double kilometer3 = kilometer * kilometer * kilometer;

constexpr double parsec = 3.0856775807e+16 * meter;

constexpr double micrometer = 1.e-6 * meter;
constexpr double nanometer  = 1.e-9 * meter;
constexpr double angstrom   = 1.e-10 * meter;
constexpr double fermi      = 1.e-15 * meter;

constexpr double barn      = 1.e-28 * meter2;
constexpr double millibarn = 1.e-3 * barn;
constexpr double microbarn = 1.e-6 * barn;
constexpr double nanobarn  = 1.e-9 * barn;
constexpr double picobarn  = 1.e-12 * barn;

// volume
constexpr double liter = 1.e+3 * centimeter3;
constexpr double dL    = 1.e-1 * liter;
constexpr double cL    = 1.e-2 * liter;
constexpr double mL    = 1.e-3 * liter;

// symbols
constexpr double nm = nanometer;
constexpr double um = micrometer;

constexpr double mm  = millimeter;
constexpr double mm2 = millimeter2;
constexpr double mm3 = millimeter3;

constexpr double cm  = centimeter;
constexpr double cm2 = centimeter2;
constexpr double cm3 = centimeter3;

constexpr double m  = meter;
constexpr double m2 = meter2;
constexpr double m3 = meter3;

constexpr double km  = kilometer;
constexpr double km2 = kilometer2;
constexpr double km3 = kilometer3;

constexpr double pc = parsec;

constexpr double L = liter;

//
// Angle
//
constexpr double radian      = 1.;
constexpr double milliradian = 1.e-3 * radian;
constexpr double degree      = (3.14159265358979323846 / 180.0) * radian;

constexpr double steradian = 1.;

// symbols
constexpr double rad  = radian;
constexpr double mrad = milliradian;
constexpr double sr   = steradian;
constexpr double deg  = degree;

//
// Time [T]
//
constexpr double nanosecond  = 1.;
constexpr double second      = 1.e+9 * nanosecond;
constexpr double millisecond = 1.e-3 * second;
constexpr double microsecond = 1.e-6 * second;
constexpr double us          = microsecond;
constexpr double picosecond  = 1.e-12 * second;
constexpr double ps          = picosecond;
constexpr double femtosecond = 1.e-15 * second;

constexpr double hertz     = 1. / second;
constexpr double kilohertz = 1.e+3 * hertz;
constexpr double megahertz = 1.e+6 * hertz;

// symbols
constexpr double ns = nanosecond;
constexpr double s  = second;
constexpr double ms = millisecond;

//
// Electric charge [Q]
//
constexpr double eplus   = 1.;              // positron charge
constexpr double e_SI    = 1.602176487e-19; // positron charge in coulomb
constexpr double coulomb = eplus / e_SI;    // coulomb = 6.24150 e+18 * eplus

//
// Energy [E]
//
constexpr double megaelectronvolt = 1.;
constexpr double electronvolt     = 1.e-6 * megaelectronvolt;
constexpr double kiloelectronvolt = 1.e-3 * megaelectronvolt;
constexpr double gigaelectronvolt = 1.e+3 * megaelectronvolt;
constexpr double teraelectronvolt = 1.e+6 * megaelectronvolt;
constexpr double petaelectronvolt = 1.e+9 * megaelectronvolt;

constexpr double joule = electronvolt / e_SI; // joule = 6.24150 e+12 * MeV

// symbols
constexpr double MeV = megaelectronvolt;
constexpr double eV  = electronvolt;
constexpr double keV = kiloelectronvolt;
constexpr double GeV = gigaelectronvolt;
constexpr double TeV = teraelectronvolt;
constexpr double PeV = petaelectronvolt;

//
// Mass [E][T^2][L^-2]
//
constexpr double kilogram  = joule * second * second / (meter * meter);
constexpr double gram      = 1.e-3 * kilogram;
constexpr double milligram = 1.e-3 * gram;

// symbols
constexpr double kg = kilogram;
constexpr double g  = gram;
constexpr double mg = milligram;

//
// Power [E][T^-1]
//
constexpr double watt = joule / second; // watt = 6.24150 e+3 * MeV/ns

//
// Force [E][L^-1]
//
constexpr double newton = joule / meter; // newton = 6.24150 e+9 * MeV/mm

//
// Pressure [E][L^-3]
//
constexpr double Pa         = newton / m2; // pascal = 6.24150 e+3 * MeV/mm3
constexpr double hep_pascal = Pa;          // to match CLHEP
constexpr double bar        = 100000 * Pa; // bar    = 6.24150 e+8 * MeV/mm3
constexpr double atmosphere = 101325 * Pa; // atm    = 6.32420 e+8 * MeV/mm3

//
// Electric current [Q][T^-1]
//
constexpr double ampere      = coulomb / second; // ampere = 6.24150 e+9 * eplus/ns
constexpr double milliampere = 1.e-3 * ampere;
constexpr double microampere = 1.e-6 * ampere;
constexpr double nanoampere  = 1.e-9 * ampere;

//
// Electric potential [E][Q^-1]
//
constexpr double megavolt = megaelectronvolt / eplus;
constexpr double kilovolt = 1.e-3 * megavolt;
constexpr double volt     = 1.e-6 * megavolt;

//
// Electric resistance [E][T][Q^-2]
//
constexpr double ohm = volt / ampere; // ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

//
// Electric capacitance [Q^2][E^-1]
//
constexpr double farad      = coulomb / volt; // farad = 6.24150e+24 * eplus/Megavolt
constexpr double millifarad = 1.e-3 * farad;
constexpr double microfarad = 1.e-6 * farad;
constexpr double nanofarad  = 1.e-9 * farad;
constexpr double picofarad  = 1.e-12 * farad;

//
// Magnetic Flux [T][E][Q^-1]
//
constexpr double weber = volt * second; // weber = 1000*megavolt*ns

//
// Magnetic Field [T][E][Q^-1][L^-2]
//
constexpr double tesla = volt * second / meter2; // tesla =0.001*megavolt*ns/mm2

constexpr double gauss     = 1.e-4 * tesla;
constexpr double kilogauss = 1.e-1 * tesla;

//
// Inductance [T^2][E][Q^-2]
//
constexpr double henry = weber / ampere; // henry = 1.60217e-7*MeV*(ns/eplus)**2

//
// Temperature
//
constexpr double kelvin = 1.;

//
// Amount of substance
//
constexpr double mole = 1.;

//
// Activity [T^-1]
//
constexpr double becquerel     = 1. / second;
constexpr double curie         = 3.7e+10 * becquerel;
constexpr double kilobecquerel = 1.e+3 * becquerel;
constexpr double megabecquerel = 1.e+6 * becquerel;
constexpr double gigabecquerel = 1.e+9 * becquerel;
constexpr double millicurie    = 1.e-3 * curie;
constexpr double microcurie    = 1.e-6 * curie;
constexpr double Bq            = becquerel;
constexpr double kBq           = kilobecquerel;
constexpr double MBq           = megabecquerel;
constexpr double GBq           = gigabecquerel;
constexpr double Ci            = curie;
constexpr double mCi           = millicurie;
constexpr double uCi           = microcurie;

//
// Absorbed dose [L^2][T^-2]
//
constexpr double gray      = joule / kilogram;
constexpr double kilogray  = 1.e+3 * gray;
constexpr double milligray = 1.e-3 * gray;
constexpr double microgray = 1.e-6 * gray;

//
// Luminous intensity [I]
//
constexpr double candela = 1.;

//
// Luminous flux [I]
//
constexpr double lumen = candela * steradian;

//
// Illuminance [I][L^-2]
//
constexpr double lux = lumen / meter2;

//
// Miscellaneous
//
constexpr double perCent     = 0.01;
constexpr double perThousand = 0.001;
constexpr double perMillion  = 0.000001;
}; // namespace my_units

struct props {
  // Custom Props
  bool is_debug;
  bool is_verbose;

  bool m_splitCluster;
  double m_minClusterHitEdep;
  double m_minClusterCenterEdep;

  // neighbour checking distances
  double m_sectorDist;
  std::vector<double> u_localDistXY;
  std::vector<double> u_localDistXZ;
  std::vector<double> u_localDistYZ;
  std::vector<double> u_globalDistRPhi;
  std::vector<double> u_globalDistEtaPhi;
  std::vector<double> u_dimScaledLocalDistXY;
};

namespace Jug::Reco {

class CalorimeterIslandCluster {

private:

  struct props& p;

  // Unpack struct props
  bool m_splitCluster           = p.m_splitCluster;
  double m_minClusterHitEdep    = p.m_minClusterHitEdep;
  double m_minClusterCenterEdep = p.m_minClusterCenterEdep;
  const eicd::CalorimeterHitCollection& m_inputHitCollection;
  eicd::ProtoClusterCollection m_outputProtoCollection;
  double m_sectorDist                        = p.m_sectorDist;
  std::vector<double> u_localDistXY          = p.u_localDistXY;
  std::vector<double> u_localDistXZ          = p.u_localDistXZ;
  std::vector<double> u_localDistYZ          = p.u_localDistYZ;
  std::vector<double> u_globalDistRPhi       = p.u_globalDistRPhi;
  std::vector<double> u_globalDistEtaPhi     = p.u_globalDistEtaPhi;
  std::vector<double> u_dimScaledLocalDistXY = p.u_dimScaledLocalDistXY;

  // neighbor checking function
  std::function<eicd::Vector2f(const eicd::CalorimeterHit&, const eicd::CalorimeterHit&)> hitsDist;

  // unitless counterparts of the input parameters
  double minClusterHitEdep{0}, minClusterCenterEdep{0}, sectorDist{0};
  std::array<double, 2> neighbourDist = {0., 0.};

public:

  CalorimeterIslandCluster(const std::string& name, struct props& props, const eicd::CalorimeterHitCollection& chc)
      : p(props), m_inputHitCollection(chc) {}

  int initialize();

  eicd::ProtoClusterCollection execute();

private:
  bool is_neighbour(const eicd::CalorimeterHit&, const eicd::CalorimeterHit&) const;

  void dfs_group(std::vector<std::pair<uint32_t, eicd::CalorimeterHit>>&, int, const eicd::CalorimeterHitCollection&,
                 std::vector<bool>&) const;

  std::vector<eicd::CalorimeterHit> find_maxima(const std::vector<std::pair<uint32_t, eicd::CalorimeterHit>>&,
                                                bool) const;

  inline static void vec_normalize(std::vector<double>&);

  void split_group(std::vector<std::pair<uint32_t, eicd::CalorimeterHit>>&, const std::vector<eicd::CalorimeterHit>&,
                   eicd::ProtoClusterCollection&) const;

}; // class


} // namespace Jug::Reco
