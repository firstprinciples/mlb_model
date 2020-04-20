mlb_model
==============================

MLB modeling project

FirstPrinciples Data Science Project Organization
------------

```
.
├── README.md                   <-- This README
├── apeel_mlb_model  <-- package, named `apeel_<repo-name>` to avoid PyPI naming conflicts
│   ├── __init__.py             <-- Specify version number of package. <major>.<minor>.<build>
│   │                               Increments correspond to <API change>.<Feature addition>.<Improvement>    
│   ├── data                    <-- All data files
│   │   ├── external            <-- Outside data goes here
│   │   ├── raw                 <-- raw, immutable data file
│   │   └── transformed         <-- transformed, merged, etc version of raw data in `transformed.csv`
│   ├── figures                 <-- .png etc files.
│   ├── notebooks               <-- Jupyter notebooks, including EDA and formal reports.
│   ├── predict                 <-- API code for trained models. Use base class `TrainedModel`.
│   │   ├── __init__.py 
│   │   └── predict.py          
│   ├── train                   <-- Transform raw data and use transformed data to train models          
│   │   ├── __init__.py
│   │   ├── train.py            <-- Code that uses data in `transformed.csv` to train ML models and serialize them
│   │   │                           in `trained_models`. Use base class `ModelTrainer`. 
│   │   └── preprocess.py        <-- Transform data using DataTransformer
│   │                           <-- Optional: if needed, add module `dataset.py` to make/query dataset
│   └── trained_models          <-- Serialized trained models.
├── requirements.txt            <-- `pip freeze > requirements.txt` your vrtl env so that results can be reproduced. 
├── setup.py                    <-- make package installable with `pip install -e` and define `install_requirements`
│                                   `setup.py` = abstract requirements,  requirements.txt => concrete (versioned)
└── unit_tests
    └── test_environment.py
```
--------

Modifiers

AP    appeal play
BP    pop up bunt
BG    ground ball bunt
BGDP  bunt grounded into double play
BINT  batter interference
BL    line drive bunt
BOOT  batting out of turn
BP    bunt pop up
BPDP  bunt popped into double play
BR    runner hit by batted ball
C     called third strike
COUB  courtesy batter
COUF  courtesy fielder
COUR  courtesy runner
DP    unspecified double play
E$    error on $
F     fly
FDP   fly ball double play
FINT  fan interference
FL    foul
FO    force out
G     ground ball
GDP   ground ball double play
GTP   ground ball triple play
IF    infield fly rule
INT   interference
IPHR  inside the park home run
L     line drive
LDP   lined into double play
LTP   lined into triple play
MREV  manager challenge of call on the field
NDP   no double play credited for this play
OBS   obstruction (fielder obstructing a runner)
P     pop fly
PASS  a runner passed another runner and was called out
R$    relay throw from the initial fielder to $ with no out made
RINT  runner interference
SF    sacrifice fly
SH    sacrifice hit (bunt)
TH    throw
TH%   throw to base %
TP    unspecified triple play
UINT  umpire interference
UREV  umpire review of call on the field

Event types:

          0    Unknown event
          1    No event
          2    Generic out
          3    Strikeout
          4    Stolen base
          5    Defensive indifference
          6    Caught stealing
          7    Pickoff error
          8    Pickoff
          9    Wild pitch
          10   Passed ball
          11   Balk
          12   Other advance
          13   Foul error
          14   Walk
          15   Intentional walk
          16   Hit by pitch
          17   Interference
          18   Error
          19   Fielder's choice
          20   Single
          21   Double
          22   Triple
          23   Home run
          24   Missing play

Batter Event Descriptors:

$: Fly ball out to fielder
$$: Ground out code
($): Force out
$(%)$ or $$(%)$: Double play codes (% indicates runner original location)
$(B)$(%): Lined into double play
C: Interference
S$: single ($ indicates fielder first touching the ball)
D$: double
T$: triple
DGR: ground rule double
E$: error
FC$: Fielder's choice
FLE$ Error on foul fly ball
H or HR: Home run
H$ or HR$: Inside-the-park home run
HP: Hit by pitch
K: Strike out
K+event: Strike out + base running event
I or IW: Intentional walk
W: Walk

Base running events:

BK: balk
CS%($$): Caught stealing
DI: Defensive indifference
OA: Other advances
PB: Passed ball
WP: Wild pitch
PO%($$): Picked off base %
POCS%($$): Picked off caught stealing
SB%: Stolen base

Pitch sequence descriptors:
 
    +  following pickoff throw by the catcher
    *  indicates the following pitch was blocked by the catcher
    .  marker for play not involving the batter
    1  pickoff throw to first
    2  pickoff throw to second
    3  pickoff throw to third
    >  Indicates a runner going on the pitch

    B  ball
    C  called strike
    F  foul
    H  hit batter
    I  intentional ball
    K  strike (unknown type)
    L  foul bunt
    M  missed bunt attempt
    N  no pitch (on balks and interference calls)
    O  foul tip on bunt
    P  pitchout
    Q  swinging on pitchout
    R  foul ball on pitchout
    S  swinging strike
    T  foul tip
    U  unknown or missed pitch
    V  called ball because pitcher went to his mouth
    X  ball put into play by batter
    Y  ball put into play on pitchout

Data columns:

"gameid",
"opp",
"inning",
"batting team",
"outs",
"balls",
"strikes",
"pitch sequence",
"visiting_score",
"home_score",
"res batter",
"res batter hand",
"res pitcher",
"res pitcher hand",
"1st runner",
"2nd runner",
"3rd runner",
"event text",
"leadoff flag",
"pinchhit flag",
"defensive position",
"lineup position",
"event type",
"batter event flag",
"ab flag",
"hit value",
"sacrifice hit flag",
"sacrifice fly flag",
"outs on play",
"RBI on play",
"wild pitch flag",
"passed ball flag",
"num errors",
"batter dest",
"runner on 1st dest",
"runner on 2nd dest",
"runner on 3rd dest"]