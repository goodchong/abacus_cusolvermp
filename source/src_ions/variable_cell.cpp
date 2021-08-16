#include "variable_cell.h"
#include "../src_pw/global.h"
#include "../input.h"

Variable_Cell::Variable_Cell(){}
Variable_Cell::~Variable_Cell(){}

void Variable_Cell::init_after_vc(void)
{
	TITLE("Variable_Cell","init_after_vc");

    GlobalC::ucell.setup_cell_after_vc(GlobalV::global_pseudo_dir, GlobalC::out, GlobalV::global_atom_card, GlobalV::ofs_running);
    DONE(GlobalV::ofs_running, "SETUP UNITCELL");

    if(ModuleSymmetry::Symmetry::symm_flag)
    {
        GlobalC::symm.analy_sys(GlobalC::ucell, GlobalC::out, GlobalV::ofs_running);
        DONE(GlobalV::ofs_running, "SYMMETRY");
    }

    GlobalC::kv.set_after_vc(GlobalC::symm, GlobalV::global_kpoint_card, GlobalV::NSPIN, GlobalC::ucell.G, GlobalC::ucell.latvec);
    DONE(GlobalV::ofs_running, "INIT K-POINTS");

    GlobalC::pw.update_gvectors(GlobalV::ofs_running, GlobalC::ucell);

    GlobalC::pw.setup_structure_factor();

    if(GlobalV::BASIS_TYPE=="pw")
    {
        GlobalC::wf.init_after_vc(GlobalC::kv.nks);
        GlobalC::wf.init_at_1();
    }

    GlobalV::ofs_running << " Setup the Vl+Vh+Vxc according to new structure factor and new charge." << endl;
    //=================================
    // initalize local pseudopotential
    //=================================
    GlobalC::ppcell.init_vloc(GlobalC::pw.nggm, GlobalC::ppcell.vloc);
    DONE(GlobalV::ofs_running,"LOCAL POTENTIAL");

    //======================================
    // Initalize non local pseudopotential
    //======================================
    if(GlobalV::BASIS_TYPE=="pw")
    {
        GlobalC::ppcell.init_vnl(GlobalC::ucell);
        DONE(GlobalV::ofs_running,"NON-LOCAL POTENTIAL");
    }

    return;
}
    

void Variable_Cell::final_calculation_after_vc(void)
{
	TITLE("Variable_Cell","final_after_vc");

    cout<<" -----------------------------------------------------------------"<<endl;

    cout<<"\n -----------------------------------------------------------------"<<endl;
    cout<<" The structure has been fully relaxed or MD finished, and the following is a scf"<<endl;
    cout<<" calculation at the final structure. The fft grids and G-vectors "<<endl;
    cout<<" are recalculated for the final relaxed unit cell."<<endl;
    cout<<" -----------------------------------------------------------------"<<endl;

    OUT(GlobalV::ofs_running," ------------------------------------------------------------------------------------");

    OUT(GlobalV::ofs_running,"\n ------------------------------------------------------------------------------------");
    OUT(GlobalV::ofs_running," The structure has been fully relaxed, and the following is a scf calculation");
    OUT(GlobalV::ofs_running," at the final structure. The fft grids and G-vectors are recalculated for the");
    OUT(GlobalV::ofs_running," final relaxed unit cell.");
    OUT(GlobalV::ofs_running," ------------------------------------------------------------------------------------");

    // (5) Setup the unitcell.
    GlobalC::ucell.setup_cell_after_vc(GlobalV::global_pseudo_dir, GlobalC::out, GlobalV::global_atom_card, GlobalV::ofs_running);
    DONE(GlobalV::ofs_running, "SETUP UNITCELL");

    // (6) symmetry analysize.
    if(ModuleSymmetry::Symmetry::symm_flag)
    {
        GlobalC::symm.analy_sys(GlobalC::ucell, GlobalC::out, GlobalV::ofs_running);
        DONE(GlobalV::ofs_running, "SYMMETRY");
    }


    // (7) Setup the k points according to symmetry.
    GlobalC::kv.set( GlobalC::symm, GlobalV::global_kpoint_card, GlobalV::NSPIN, GlobalC::ucell.G, GlobalC::ucell.latvec );
    DONE(GlobalV::ofs_running,"INIT K-POINTS");

    // (1) Init the plane wave.
    GlobalC::pw.gen_pw(GlobalV::ofs_running, GlobalC::ucell, GlobalC::kv);
    DONE(GlobalV::ofs_running,"INIT PLANEWAVE");
    cout << " UNIFORM GRID DIM     : " << GlobalC::pw.nx <<" * " << GlobalC::pw.ny <<" * "<< GlobalC::pw.nz << endl;
    cout << " UNIFORM GRID DIM(BIG): " << GlobalC::pw.nbx <<" * " << GlobalC::pw.nby <<" * "<< GlobalC::pw.nbz << endl;

    // init the grid, then the charge
    // on grid can be distributed.
    GlobalC::Pgrid.init_final_scf(GlobalC::pw.ncx, GlobalC::pw.ncy, GlobalC::pw.ncz, 
		GlobalC::pw.nczp, GlobalC::pw.nrxx, GlobalC::pw.nbz, GlobalC::pw.bz); // mohan add 2010-07-22, update 2011-05-04

    //=====================
    // init potential
    //=====================
    GlobalC::CHR.init_final_scf();
    GlobalC::pot.allocate(GlobalC::pw.nrxx);

    //=====================
    // init wave functions
    //=====================
    if(GlobalV::BASIS_TYPE=="pw")
    {
        GlobalC::wf.allocate(GlobalC::kv.nks);
    }
    else
    {
        GlobalC::wf.allocate_ekb_wg(GlobalC::kv.nks);
    }
    GlobalC::UFFT.allocate();

    //=======================
    // init pseudopotential
    //=======================
    if(GlobalV::BASIS_TYPE=="pw")
	{
		GlobalC::ppcell.init(GlobalC::ucell.ntype);
	}

    //=====================
    // init hamiltonian
    //=====================
    GlobalC::hm.hpw.allocate(GlobalC::wf.npwx, GlobalV::NPOL, GlobalC::ppcell.nkb, GlobalC::pw.nrxx);
    GlobalC::hm.hpw_gpu.allocate(GlobalC::wf.npwx, GlobalV::NPOL, GlobalC::ppcell.nkb, GlobalC::pw.nrxx);

    //=================================
    // initalize local pseudopotential
    //=================================
    GlobalC::ppcell.init_vloc(GlobalC::pw.nggm, GlobalC::ppcell.vloc);
    DONE(GlobalV::ofs_running,"LOCAL POTENTIAL");
    //======================================
    // Initalize non local pseudopotential
    //======================================
    if(GlobalV::BASIS_TYPE=="pw")
    {
        GlobalC::ppcell.init_vnl(GlobalC::ucell);
        DONE(GlobalV::ofs_running,"NON-LOCAL POTENTIAL");
    }
    //=========================================================
    // calculate the total local pseudopotential in real space
    //=========================================================
    GlobalC::pot.init_pot(0, GlobalC::pw.strucFac);//atomic_rho, v_of_rho, set_vrs

    //==================================================
    // Create GlobalC::ppcell.tab_at , for trial wave functions.
    // Initial start wave functions
    //================================
    if(GlobalV::BASIS_TYPE=="pw")
    {
		GlobalC::pot.newd();

		GlobalC::wf.init_at_1();

        GlobalC::wf.wfcinit();
        DONE(GlobalV::ofs_running,"INIT BASIS");
    }

	return;
}
