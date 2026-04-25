
# from svgutils.compose import *
# import os
#

#
# Figure("16cm", "6.5cm",
#         Panel(
#               SVG("%s.svg" % fileName),
#               #Text("A", 25, 20, size=12, weight='bold')
#              ),
#         Panel(
#               SVG("Swing_Stance_indecisive.svg").scale(0.6),
#               #Text("B", 25, 20, size=12, weight='bold')
#              ).move(280, 0)
#         ).save("%s_compose.svg" % fileName)
#

#os.system('inkscape %s_compose.svg --export-area-drawing --batch-process --export-type=pdf --export-filename=%s_compose.pdf' % (fileName,fileName))

import svgStack as svgs
import svgutils.transform as sg
import os

fileName = 'fig_experiment-PawPositionExtraction_v0.1'
doc = svgs.Document()

layout1 = svgs.HBoxLayout()
layout1.addSVG('%s.svg' % fileName,alignment=svgs.AlignTop|svgs.AlignHCenter)
#layout1.addSVGNoLayout('protocolIllustration.svg',x=0,y=-100)
#layout1.addSVGNoLayout('RunningWheel5-Wheel.svg',x=-200,y=0)
#layout1.addSVG('RunningWheel5-Wheel.svg',alignment=svgs.AlignCenter)
layout1.addSVGNoLayout('stance-swing-miss.svg',x=-500,y=0)
#layout1.addSVGNoLayout('circle-cyan.svg',x=-500,y=0)
layout1.addSVGNoLayout('blue_triangle2.svg',x=-500,y=0)

#layout1.addSVGNoLayout('fig_modelPredictionBurstsIndividual_VenancesBin0.svg',x=00,y=00)
#layout1.addSVGNoLayout('spatial_illustration.svg',x=-1450,y=-35)

#layout2 = ss.VBoxLayout()

#layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
#layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
#layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
#layout1.addLayout(layout2)

doc.setLayout(layout1)

fignameCompose = '%s_compose' % fileName

doc.save('%s.svg' % fignameCompose)
#os.system('inkscape %s.svg --export-type=pdf --export-filename=%s.pdf' %(fignameCompose,fignameCompose) )
#os.system('convert '+str(fignameCompose)+'.pdf -quality 200 '+str(fignameCompose)+'.png')
