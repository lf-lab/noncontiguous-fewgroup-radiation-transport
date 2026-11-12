# Non-contiguous few-group radiation transport: demonstration code

Builds a K-group opacity table (default K = 150) from a 1500-group table by the
non-contiguous generalized-group method, and tests it with 1D multigroup
diffusion in an isothermally expanding plasma.

**The opacity table here is a dummy** (fictitious element, random numbers, no
physical meaning). The tin table used in the paper cannot be distributed.

```
pip install -r requirements.txt
python3 step1_make_dummy_opacity.py     # dummy 1500-group table
python3 step2_build_fewgroup_table.py   # training runs + 150-group table
python3 step3_test_fewgroup_table.py    # test on unseen cases + figures
```

Full instructions: open `README.html` in a browser.
