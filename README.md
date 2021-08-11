# JetCLR
Core codebase for **JetCLR**, a high-energy-physics tool for the self-supervised contrastive learning of jet representations.

JetCLR uses a permutation-invariant transformer-encoder network and a contrastive loss function to map jet constituents to a representation space which is approximately invariant to a set of symmetries and augmentations of the jet data, and is discriminant within the dataset it is optimised on.  The symmetries and augmentations are coded in *scripts/modules/jet_augs.py*, they are: <br />
* **symmetries**:
  * rotations in the rapidity-azimuthal plane, around the transverse-momentum-weighted centroid of the jet
  * translations in the rapidity-azimuthal plane
  * permutation invariance of the jet constituents, this is ensured by the network architecture
* **augmentations**:
  * smearing of constituent coordinates, inversely proportional to their transverse momentum
  * collinear splittings of jet constituents

The scheme for optimising the network is inspired by the SimCLR<sup>[1](#myfootnote1)</sup> paper, and is coded here in *scripts/run_jetclr.py*.
The mapping to the new representation space is entirely self-supervised, using only the physically-motivated invariances to transformations and augmentations of the data.  Truth labels are not used in the optimisation of the JetCLR network.

For questions/comments about the code contact: dillon@thphys.uni-heidelberg.de

---

This code was initially written for the paper:

**Symmetries, Safety, and Self-Supervision**<br />
https://arxiv.org/abs/2108.04253 <br />
*Barry M. Dillon, Gregor Kasieczka, Hans Olischlager, Tilman Plehn, Peter Sorrenson, and Lorenz Vogel*

---
<sub> <a name="myfootnote1">1</a>: 'A Simple Framework for Contrastive Learning of Visual Representations', Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton (arxiv:2002.05709) </sub>
