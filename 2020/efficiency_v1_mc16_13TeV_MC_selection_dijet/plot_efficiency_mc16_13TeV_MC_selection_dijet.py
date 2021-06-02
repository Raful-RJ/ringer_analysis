from Gaugi.messenger import LoggingLevel, Logger
from EfficiencyTools import PlotProfiles
from Gaugi.storage import  restoreStoreGate
from Gaugi.monet import AtlasTemplate1, SetAtlasStyle, setLegend1,  AddTexLabel
from ROOT import gROOT, TLegend
from ROOT import kBlack, kBlue, kGray, kOrange, kPink, kViolet, kRed, kCyan, kAzure
from ROOT import kPlus, kCircle, kMultiply, kStar, kDot, kOpenTriangleUp, kFullTriangleDown, kFullSquare, kFullCircle
from ROOT import kBlack, kBlue, kRed, kCircle, kFullCircle, kOpenSquare, kFullSquare, kOpenTriangleUp, kFullTriangleUp, kMultiply

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
triggers = [
            # "EMU_g20_tight_noringer_L1EM15VHI",
            # "EMU_g20_tight_ringer_v1_L1EM15VHI",
            # "EMU_g20_medium_noringer_L1EM15VHI",
            # "EMU_g20_medium_ringer_v1_L1EM15VHI",
            # "EMU_g20_loose_noringer_L1EM15VHI",
            # "EMU_g20_loose_ringer_v1_L1EM15VHI",
            "EMU_g35_tight_noringer_L1EM22VHI",
            "EMU_g35_tight_ringer_v1_L1EM22VHI",
            "EMU_g35_medium_noringer_L1EM22VHI",
            "EMU_g35_medium_ringer_v1_L1EM22VHI",
            "EMU_g35_loose_noringer_L1EM22VHI",
            "EMU_g35_loose_ringer_v1_L1EM22VHI",
            # "EMU_g25_tight_noringer_L1EM20VHI",
            # "EMU_g25_tight_ringer_v1_L1EM20VHI",
            # "EMU_g25_medium_noringer_L1EM20VHI",
            # "EMU_g25_medium_ringer_v1_L1EM20VHI",
            # "EMU_g25_loose_noringer_L1EM20VHI",
            # "EMU_g25_loose_ringer_v1_L1EM20VHI",
]

theseColors = [kBlack, kBlack, kRed, kRed, kBlue, kBlue]
theseMarkers = [kCircle, kFullCircle, kOpenSquare, kFullSquare, kOpenTriangleUp, kFullTriangleUp]
# theseMarkers = [kCircle, kMultiply, kOpenSquare, kMultiply, kOpenTriangleUp, kMultiply]

theseTransColors = [i+1 for i in range(0,len(triggers))]
eff_et  = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_et' ) for trigger in triggers ]
eff_eta = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_eta' ) for trigger in triggers ]
eff_phi = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_phi' ) for trigger in triggers ]
eff_mu  = [ sg.histogram( basepath+'/'+trigger+'/Efficiency/' + level + '/eff_mu' ) for trigger in triggers ]

efficiencyObjects = { '\rE_{T}[GeV]'   : eff_et,
                      '#eta'      : eff_eta, 
                      '#varphi'   : eff_phi, 
                      '<#mu>'     : eff_mu
                      }
for hist in eff_mu:
  hist.SetBins(9,0,40);  
SetAtlasStyle()

legend =TLegend(0.2,0.5,0.5,0.7)
legends=[]
for idx, hist in enumerate(eff_et):
    temp_legend = triggers[idx].replace('EMU','HLT')
    try:
        legends.append(temp_legend.replace('_noringer',''))
    except:
        legends.append(temp_legend.replace('_ringer_v1',''))


for idx, hist in enumerate(eff_et):
  hist.SetMarkerStyle(theseMarkers[idx])
  if 'ringer' in legends[idx]: continue
  legend.AddEntry(hist, legends[idx].replace('EMU','HLT'),'l')

fileName = ['fakeEt', 'fakelEta', 'fakePhi', 'fakeMu']

for idx, obj in enumerate(efficiencyObjects):
    eff_canvas = PlotProfiles( efficiencyObjects[obj], xlabel=obj, these_colors = theseColors, these_transcolors = theseTransColors, these_markers = theseMarkers, ylabel='Trigger Background Efficiency'  ,y_axes_maximum=0.6)
    setLegend1(legend)
    AtlasTemplate1(canvas = eff_canvas, dolumi=True, atlaslabel='Internal, MC data 2016')
    AddTexLabel(eff_canvas, 0.2, 0.75, 'close: with Ringer open: without Ringer', textsize=0.04)
    eff_canvas.SaveAs(level + '_' + fileName[idx] + '.pdf')

for trigger in triggers:
  plot_table( sg, mainLogger, trigger, basepath )
