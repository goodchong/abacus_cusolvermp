#include "pw_basis.h"

PW_Basis::PW_Basis()
{
    ig2isz = NULL;
    istot2ixy = NULL;  
    ixy2istot = NULL;
    is2ixy = NULL;
    ixy2ip = NULL; 
    startnsz_per = NULL;
    nstnz_per = NULL;
    gdirect = NULL;		
    gcar = NULL; 
    gg = NULL;
    startz = NULL;
    numz = NULL; 
}
PW_Basis:: ~PW_Basis()
{
    if(ig2isz != NULL) delete[] ig2isz;
    if(istot2ixy != NULL) delete[] istot2ixy;
    if(ixy2istot != NULL) delete[] ixy2istot;
    if(is2ixy != NULL) delete[] is2ixy;
    if(ixy2ip != NULL) delete[] ixy2ip;
    if(startnsz_per != NULL) delete[] startnsz_per;
    if(nstnz_per != NULL) delete[] nstnz_per;
    if(gdirect != NULL) delete[] gdirect;
    if(gcar != NULL) delete[] gcar;
    if(gg != NULL) delete[] gg;
    if(startz != NULL) delete[] startz;
    if(numz != NULL) delete[] numz;
}