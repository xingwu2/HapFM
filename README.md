# HapFM

HapFM has 2 main functions: genome-wide haplotype block partition and haplotype-based fine-mapping.

## Installation

HapFM can be install both manually and via conda/mamba package managers. 

### Install via conda/mamba (recommended because it is one push button)

```mamba env create -f environment.yml```  
```mamba activate hapfm```

### Install mannually (if you have the patience and want to practice installing softwares)

Please first ensure python3 and R are in your environment path. Then install these modules/libraries

python3: numpy,scipy,pandas,networkx,pyclustering and scikit-learn

R: gpart and optparse

## Usage

### Haplotype block partition and cluster formation
The default command line for the step is  
```python3 bin/HapFM_haplotype.py -v VCF -b bigld -o output```

The output file (haplotypeDM) will become the input of the haplotype-based fine-mapping step.

### Haplotype-based fine-mapping
The default command line for the step is  
```python3 bin/HapFM_mapping.py -i haplotypeDM -c covariates -y phenotype -o output```


## Citation
Wu X, Jiang W, Fragoso C, Huang J, Zhou G, Zhao H, et al. (2022) Prioritized candidate causal haplotype blocks in plant genome-wide association studies. PLoS Genet 18(10): e1010437. https://doi.org/10.1371/journal.pgen.1010437




