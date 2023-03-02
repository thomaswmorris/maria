SCANNINGPATTERNS = {
    'static': {
        'start_time'    : '2022-07-01T08:00:00',
        'end_time'      : '2022-07-01T08:10:00',
        'scan_pattern'  : 'back-and-forth',
        'scan_options'  : {'k' : 3.1416},     # 
        'coord_center'  : (45, 45),
        'coord_throw'   : (2, 0),
        'coord_frame'   : 'az_el',           # or ra_dec
        'scan_period'   : 120,                 # [s]   how often the scan pattern repeats
        'sample_rate'   : 20,                  # [Hz]  how fast to sample
    },

    'box': {
        ...    
    },

    'daisy': {
        'start_time'   : '2022-07-01T08:00:00',
        'end_time'     : '2022-07-01T08:10:00',
        'scan_pattern' : 'daisy',          # [.]   the type of scan strategy (SS)
        'scan_options' : {'k' : 3.1416},   # 
        'coord_center' : (45, 45),
        'coord_throw'  : (2, 2),
        'coord_frame'  : 'az_el',
        'scan_period'  : 120,              # [s]   how often the scan pattern repeats
        'sample_rate'  : 20,               # [Hz]  how fast to sample
    },

    'lissajous': {
        ...
    }
}
