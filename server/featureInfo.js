const FEATURE_INFO = {
    DH: {
      label: 'DH: Disc Hemorrhage',
      definition:
        'Splinter- or flame-shaped hemorrhage within one disc diameter of the optic-nerve-head margin.',
      significance:
        'Sensitive marker of active glaucomatous damage that predicts faster RNFL thinning.'
    },
    RNFLDS: {
      label: 'RNFLDS: Retinal Nerve-Fiber-Layer Defect (Superior)',
      definition:
        'Wedge-shaped dark gap in the superior peripapillary RNFL on red-free imaging or OCT.',
      significance:
        'Indicates early structural glaucoma and corresponds to an inferior arcuate visual-field defect.'
    },
    BCLVI: {
      label: 'BCLVI: Baring Circumlinear Vessel (Inferior)',
      definition:
        'Inferior circumlinear vessel left unsupported over the optic cup because of adjacent rim loss.',
      significance:
        'Early sign of inferior rim loss and impending functional decline.'
    },
    LD: {
      label: 'LD: Laminar Dots',
      definition:
        'Visible pores of the lamina cribrosa exposed within the optic cup after tissue loss.',
      significance:
        'Signify deep cupping and are associated with moderate-to-advanced glaucomatous neuropathy.'
    },
    ANRS: {
      label: 'ANRS: Appearance Neuro-retinal Rim Thinning (Superior)',
      definition:
        'Thinning/notching of the superior rim violating the ISNT rule.',
      significance:
        'Predicts inferior visual-field defects and signals progression.'
    },
    ANRI: {
      label: 'ANRI: Appearance Neuro-retinal Rim Thinning (Inferior)',
      definition:
        'Thinning/notching of the inferior rim, often the earliest sector affected.',
      significance:
        'Correlates with superior visual-field loss and indicates early disease.'
    },
    NVT: {
      label: 'NVT: Nasalisation of Vessel Trunk',
      definition:
        'Displacement of the central retinal vessel trunk toward the nasal disc edge.',
      significance:
        'Associated with advanced cupping and rapid central field decline.'
    },
    BCLVS: {
      label: 'BCLVS: Baring Circumlinear Vessel (Superior)',
      definition:
        'Superior circumlinear vessel unsupported over the optic cup owing to rim loss.',
      significance:
        'Reflects focal superior rim loss and can precede functional loss.'
    },
    RNFLDI: {
      label: 'RNFLDI: Retinal Nerve-Fiber-Layer Defect (Inferior)',
      definition:
        'Wedge-shaped dropout of the inferior peripapillary RNFL.',
      significance:
        'Strongly predicts superior visual-field defects and future progression.'
    },
    LC: {
      label: 'LC: Large Cup',
      definition:
        'Cup occupying an abnormally large proportion of the optic disc.',
      significance:
        'Hallmark of glaucomatous neuropathy; must be distinguished from physiologic cupping.'
    }
  };
  
module.exports = FEATURE_INFO