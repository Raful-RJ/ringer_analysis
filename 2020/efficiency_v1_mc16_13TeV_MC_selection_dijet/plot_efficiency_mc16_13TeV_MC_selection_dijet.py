from Gaugi.messenger import LoggingLevel, Logger
from EfficiencyTools import PlotProfiles
from Gaugi.storage import  restoreStoreGate
from Gaugi.monet import AtlasTemplate1, SetAtlasStyle, setLegend1
from ROOT import gROOT, TLegend
from ROOT import kBlack, kBlue, kGray, kOrange, kPink, kViolet, kRed, kCyan, kAzure
from ROOT import kPlus, kCircle, kMultiply, kStar, kDot, kOpenTriangleUp, kFullTriangleDown, kFullSquare, kFullCircle
import argparse
gROOT.SetBatch(True)


mainLogger = Logger.getModuleLogger("job")
mainLogger.level = LoggingLevel.INFO


def plot_table( sg, logger, trigger, basepath ):
  triggerLevels = ['L1Calo','L2Calo','L2','EFCalo','HLT']
  logger.info( '{:-^78}'.format((' %s ')%(trigger)) ) 
  
  for trigLevel in triggerLevels:
    dirname = basepath+'/'+trigger+'/Efficiency/'+trigLevel
    total  = sg.histogram( dirname+'/eta' ).GetEntries()
    passed = sg.histogram( dirname+'/match_eta' ).GetEntries()
    eff = passed/float(total) * 100. if total>0 else 0
    eff=('%1.2f')%(eff); passed=('%d')%(passed); total=('%d')%(total)
    stroutput = '| {0:<30} | {1:<5} ({2:<5}, {3:<5}) |'.format(trigLevel,eff,passed,total)
    logger.info(stroutput)
  logger.info( '{:-^78}'.format((' %s ')%('-')))






parser = argparse.ArgumentParser(description = '', add_help = False)


parser.add_argument('-i','--inputFile', action='store', 
    dest='inputFile', required = True, nargs='+',
    help = "The input files that will be used to generate the plots")

parser.add_argument('-l','--level', action='store', 
    dest='level', required = False, default = None,
    help = "Level of teh Efficiency Analysis (L1Calo, L2Calo, L2, EFCalo, HLT")


args = parser.parse_args()
print(args)
inputFile = args.inputFile[0]
basepath  = 'Event/EfficiencyTool'
level     = args.level



sg =  restoreStoreGate( inputFile )

# triggers = [
#                 "EMU_g20_tight_noringer_L1EM3",
#                 "EMU_g20_medium_noringer_L1EM3",
#                 "EMU_g20_loose_noringer_L1EM3",
#                 "EMU_g15_tight_noringer_L1EM3",
#                 "EMU_g15_medium_noringer_L1EM3",
#                 "EMU_g15_loose_noringer_L1EM3",
#                 "EMU_g5_tight_noringer_L1EM3",
#                 "EMU_g5_medium_noringer_L1EM3",
#                 "EMU_g5_loose_noringer_L1EM3",
#                 ]


triggers = [
                "EMU_g20_loose_noringer_L1EM3",
                "EMU_g20_loose_ringer_v1_L1EM3",

                ]

theseColors = [i+1 for i in range(0,len(triggers))]
theseMarkers = [i+4 for i in range(0,len(triggers))]
theseTransColors = [i+1 for i in range(0,len(triggers))]
eff_et  = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_et' ) for trigger in triggers ]
eff_eta = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_eta' ) for trigger in triggers ]
eff_phi = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_phi' ) for trigger in triggers ]
eff_mu  = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_mu' ) for trigger in triggers ]

efficiencyObjects = { 'E_T[GeV]'  : eff_et,
                      '#eta'      : eff_eta, 
                      '#varphi'   : eff_phi, 
                      '<#mu>'     : eff_mu
                      }
for hist in eff_mu:
  hist.SetBins(9,0,40);  
SetAtlasStyle()

legend =TLegend()

for idx, hist in enumerate(eff_et):
  legend.AddEntry(hist, triggers[idx])

for obj in efficiencyObjects:
    eff_canvas = PlotProfiles( efficiencyObjects[obj], xlabel=obj, these_colors = theseColors, these_transcolors = theseTransColors, these_markers = theseMarkers)
    setLegend1(legend)
    AtlasTemplate1(canvas = eff_canvas, dolumi=True, atlaslabel='Internal')
    eff_canvas.SaveAs(level + '_' + obj + '.pdf')

for trigger in triggers:
  plot_table( sg, mainLogger, trigger, basepath )
