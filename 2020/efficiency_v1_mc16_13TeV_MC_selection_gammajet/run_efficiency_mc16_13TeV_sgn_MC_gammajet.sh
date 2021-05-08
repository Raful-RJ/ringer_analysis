BASEPATH_gammajet=/home/juan.marin/datasets/physval/user.jlieberm.mc16a_13TeV.4231*/*.root
prun_jobs.py -c "python3 job_efficiency_mc16_13TeV_sgn_MC_gammajet.py" -mt 30  -i $BASEPATH_gammajet
mkdir samples
mv *.root samples
prun_merge.py -i samples/*.root -o efficiency_mc16_13TeV_sgn_MC_gammajet.root -mt 30
