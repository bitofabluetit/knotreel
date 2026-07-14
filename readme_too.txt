README
 
You don’t have to tho x
 
Hehe so let me be formal for a second – this repository provides the code for a range of applications, all of which are built off the foundation of taking a 1d data array, transforming it into 3d space using takens delay embed  ing and some knot geometry, and from there you can derive physical forces and variables from the 3d knot,  in an aim to get a higher fidelity of data which directly leads to a more lucid analysis of said data. 
 
That’s the elevator pitch. 

I have used this principle of 1d > 3d knot for a few different applications and they are as follows:
 
audio_visualizer
(makes an mp.4 of the knot forming from an audio file, different versions but all doing similar stuffs with varying displays)
 
macro final knot 
(compares multiple recordings of the same song and shows multiple knots for bass, mids, highs, for each instance and an overlayed knot for each hz range) 
 
bunched together 
(overlays multiple recordings into a single file)
-       Buncher.py
 
audiomk2 
(deep fake catcher)
⁃            knotreel.py (live)
 
 
 
 
 
audiomk3
(elaborated audiomk2)
⁃            newknot.py newest simplest
⁃            dud.py for biggest view and pings 
 
vivell
(protein folding and drug docking)
⁃            big_boi.py (finding structure)
⁃            evolved.py (drug docking)
 
phys 
(physics simulations for solar flares/viscous fluids (nav stokes)/prime numbers)
⁃            nasa_flare.py
⁃            nav_sim_pdf.py
⁃            primes.py
 
nature knot tester 
(a selection of experiments to look at the knot formation and intertwinement of individual audio sources in an environment :))
⁃            knoture.py 
⁃            pattern_finder.py 
⁃            macro_compare.py (nature audio correlation) 
⁃            phrasing_check.py (nature phrasing) 
 
 
 
Gosh golly that’s a bunch a stuff, its been really fun working on this project, its taken me a 3 months of thunking and 1 month of really dedicated work but the code is by no means perfect, and by no means is the project complete – im not a coder, not a tech guy really I just had an idea and though a bit of trial and error and a disgusting token usage rate on ai studio, i manged to just about hobgoblin gander, my way through it all and atleast lay some foundations. And I hope people use this as such, a starting point for some productive use.
 
In terms of using this code, they all run locally in vs, literally just copy paste code, set up venv, in some cases put in a api key or a file name, run the install commands and then you can run it and get the output graphs/ai pager/knots. 
 
Im going to be working on this for the next little while to tidy it all up and some more explanation/updates but for now my priority was just making it public. I also would like to state that this goes beyond me and as such I welcome pull requests to optimize the math, clean up the scripts, or add new data sources, with open arms and an email of nonoithinkthisistheone@gmail.com for any enquiries, if an invite for a Carhartt sponsorship comes though ill know ive made it 
 
If you got any questions, ill be no help but I can try to answer/look into what ya on about no probs, timely responses may be unlikely but just know im thunking bout it. You can use the email provided just a second ago or my insta is @bitofabluetit – take ya pick. 
 
I’ve added a tip jar link too – please do not feel in any way obliged to use it, I just need some extra funding for my coffee intake. This is the link, https://buymeacoffee.com/misch
 
--
 
## Installation & Prerequisites This suite relies on a few system-level utilities in addition to Python packages. ##
 
# 1. System Requirements * 
**FFmpeg**: Required to render and mux visualizer videos. * 
 *macOS*: `brew install ffmpeg` *
 *Windows/Linux*: Install via your package manager and ensure it is in your system PATH. * 
 
**PortAudio**: Required for live microphone audio analysis (`sounddevice`). * *macOS*: `brew install portaudio` ##
 
# 2. Python Setup I recommend using a virtual environment: 
Mac:
 
Python3 -m venv venv 
 
source venv/bin/activate 
 
(window users)
 
GET A REAL LAPTOP
 
--
 
pip install -r requirements.txt
