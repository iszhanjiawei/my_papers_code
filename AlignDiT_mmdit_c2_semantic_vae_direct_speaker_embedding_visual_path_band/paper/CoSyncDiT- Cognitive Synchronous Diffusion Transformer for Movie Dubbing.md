# CoSyncDiT: Cognitive Synchronous Diffusion Transformer for Movie Dubbing

Gaoxiang Cong1,2, Liang Li1, Jiaxin Ye3, Zhedong Zhang4, Hongming Shan3, Yuankai Qi5, and Qingming Huang2

1 Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China 2 University of Chinese Academy of Sciences, Beijing, China 3 Fudan University, Shanghai, China 4 Hangzhou Dianzi University, Hangzhou, China 5 Macquarie University, Sydney, Australia

Abstract. Movie dubbing aims to synthesize speech that preserves the vocal identity of a reference audio while synchronizing with the lip movements in a target video. Existing methods fail to achieve precise lip-sync and lack naturalness due to explicit alignment at the duration level. While implicit alignment solutions have emerged, they remain susceptible to interference from the reference audio, triggering timbre and pronunciation degradation in in-the-wild scenarios. In this paper, we propose a novel flow matching-based movie dubbing framework driven by the Cognitive Synchronous Diffusion Transformer (CoSync-DiT), inspired by the cognitive process of professional actors. This architecture progressively guides the noise-to-speech generative trajectory by executing acoustic style adapting, fine-grained visual calibrating, and timeaware context aligning. Furthermore, we design the Joint Semantic and Alignment Regularization (JSAR) mechanism to simultaneously constrain frame-level temporal consistency on the contextual outputs and semantic consistency on the flow hidden states, ensuring robust alignment. Extensive experiments on both standard benchmarks and challenging in-the-wild dubbing benchmarks demonstrate that our method achieves the state-of-the-art performance across multiple metrics.

Keywords: Movie dubbing · Flow matching · Visual voice cloning

## 1 Introduction

Movie Dubbing, also known as Visual Voice Cloning (V2C) [3], aims to generate a piece of speech for the character in silent videos based on the specified reference voice and textual scripts (as shown in Fig. 1(a)). It promises significant potential in real-world applications such as film post-production, media production, and personal speech AIGC. Compared to traditional video-to-speech tasks [7,19–21], movie dubbing presents significantly greater challenges. Unlike merely inferring speech from visual cues, V2C must faithfully clone the reference acoustic timbre and accurately articulate the textual scripts aligned with dynamic lip motions in the video, while ensuring emotionally expressive and high-fidelity speech quality.

![](images/bcd991fbc425c921fc692450aa3100c8d4cb808cde945c0997ce61bc30a8cf4d.jpg)  
Fig. 1: (a) Illustration of Visual Voice Cloning (V2C) task. (b) Explicit alignment manner requires an external forced tool to compute the ground-truth duration of each phoneme in advance. (c) Implicit alignment eliminates the dependency on external forced tools. (d) We propose a movie dubbing architecture built upon CoSync-DiT, which is structured around a cognitively inspired listen–watch–articulate paradigm to progressively guide the denoising trajectory in flow matching.

Previous works mainly focus on improving pronunciation clarity and prosody modeling. For instance, Speaker2Dubber [50] introduces a two-stage architecture by pre-training on the clean TTS corpus to improve clarity. Then, ProDubber [52] employs a prosody-enhanced pre-training paradigm on the larger corpus. Recently, InstructDubber [51] integrates pre-trained TTS framework with LLM-driven emotion instructions to predict pitch and energy variations. However, these methods rely on TTS architecture and predict integer-scaled durations for each phoneme (as shown in Fig. 1(b)), failing to achieve the precise lip-sync required for dubbing. Some dubbing methods [10, 12] attempt to introduce duration-level contrastive learning between phoneme and lip sequences to improve alignment. Nevertheless, these methods still depend on an external forced tool (e.g., Montreal Forced Aligner [30]) to extract ground-truth duration boundaries for identifying positive samples during contrastive learning. This rigidly restricts the pronunciation to predefined intervals, damaging the naturalness and expressiveness of the generated speech [18].

Recently, AlignDiT [6] has represented a significant leap forward in the field by eliminating the need for external forced alignment tools and adopting an implicit alignment modeling approach. As shown in Fig. 1(c), it fuses text with complementary-masked audio and visual features via cross-attention across every layer of the diffusion transformer. However, this strategy compels the text to simultaneously align all signals, rendering the overall alignment highly susceptible to interference from the early segments of the reference audio, particularly during an in-the-wild scenario. Furthermore, uniformly applying text cross-attention across all DiT layers can complicate the alignment dynamics, potentially compromising both timbre and pronunciation clarity.

In this paper, we propose a novel flow matching-based movie dubbing framework built upon Cognitive Synchronous Diffusion Transformer (CoSync-DiT), which progressively guides a noise-to-speech generative trajectory to effectively resolve the demands of high-fidelity style cloning and exact audio-visual synchronization. Inspired by the dubbing workflow of professional actors [2], our method is structured around a sequential cognitive process of listening, watching, and articulating. Rather than treat all transformer layers homogeneously, CoSync-DiT partitions the denoising generation within flow matching into three distinct phases (as shown in Fig. 1(d)). First, we initialize a unified acousticsemantic prior. Then, the model leverages multi-head self-attention to capture the acoustic style and establish its relationship with the text, without any visual interference. Second, we inject fine-grained visual information through residual connections using a zero-initialized learnable gate. This step calibrates the previously acquired acoustic representations to capture temporal dynamics. Third, we design a time-aware context aligning layer to yield correct articulation by implicitly retrieving the linguistic context based on second phase. Finally, we introduce a Joint Semantic and Alignment Regularization (JSAR) mechanism to further rectify the third phase. This approach innovatively applies frame-level contrastive learning to the queried contextual outputs to enforce strict temporal alignment, while ensuring semantic constraints on the DiT hidden states.

Our key contributions are: (1) We propose CoSync-DiT, a novel flow matching-based framework for movie dubbing. By formulating the denoising generation as a cognitively synchronous process, it progressively guides the vector field estimation through three sequential phases: acoustic style adapting, finegrained visual calibrating, and time-aware context aligning. (2) We design the JSAR mechanism to enforce frame-level temporal consistency on intermediate context outputs and semantic consistency on final flow hidden states, thereby stabilizing the generative trajectory and enhancing context alignment. (3) Our method performs favourably against state-of-the-art methods on three public datasets, including the challenging in-the-wild movie dubbing scenarios.

## 2 Related Work

## 2.1 Visual Voice cloning

V2C [3,27,32,53] has recently attracted widespread attention and promises a profound impact on film production [26, 39, 46, 54] and digital media [8, 38, 45, 47]. V2C aims to generate synchronized speech for silent videos conditioned on a specific reference timbre. Early methods widely adopt explicit duration prediction to achieve lip synchronization. For instance, StyleDubber [13] queries lip motions using textual phonemes to construct a phoneme-level visual sequence and predicts the duration of each phoneme. ProDubber [52] and InstructDubber [51] follow a similar paradigm. By enforcing explicit duration boundaries, these approaches typically guarantee high pronunciation clarity. However, they inherently struggle to achieve precise lip synchronization because the predicted durations must be expanded into rigid integer multiples. Recent methods [10,12] introduce duration-level contrastive learning to alleviate this limitation. Nevertheless, they still rely on external forced aligners to extract duration intervals. In this paper, drawing inspiration from the human cognitive workflow, we propose CoSync-DiT, a novel implicit dubbing framework that completely eliminates the dependency on external aligners. It ensures high-fidelity timbre cloning and accurate lip synchronization, even in challenging in-the-wild dubbing scenarios.

## 2.2 Flow Matching and Generative Modeling

Flow matching [25] has emerged as a dominant paradigm in generative modeling due to its superior quality and inference speed. It learns a vector field [40] to transport samples from a simple prior distribution to the target data distribution along efficient probability paths. In TTS field, flow matching has found extensive applications [5, 14, 15, 22, 23, 31, 44, 49, 55]. However, these methods cannot parse visual scenes and fail to achieve fine-grained lip-sync, hindering their application in movie dubbing. FlowDubber [10] introduces a dual contrastive learning, which relies on external forced aligners and employs a monotonic search algorithm for sequential expansion. Nevertheless, even minor initial misalignments tend to compound throughout the generation process and degrade the lip-sync quality. AlignDiT [6] implicitly models this alignment by incorporating cross-attention modules across all transformer layers. However, due to its complementary masking strategy, the model struggles to simultaneously align with the reference audio and the target lip motions. This homogeneous injection across all layers further exacerbates alignment instability in complex dubbing scenarios. In this paper, we propose CoSync-DiT to progressively guide the denoising trajectory by executing acoustic style adapting, fine-grained visual calibrating, and time-aware context aligning. Furthermore, we design the JSAR mechanism to learn temporal consistency and semantic consistency.

## 3 Methodology

As shown in Fig. 2, the overall framework of our method includes the following parts. First, individual encoders extract the foundational representations from the reference audio, silent video, and dubbing scripts. Next, under the Optimal-Transport Conditional Flow Matching (OT-CFM) paradigm, the proposed CoSync-DiT first takes Gaussian noise, masked acoustic conditions, and semantic conditions as priors to execute Acoustic Style Adapting. Sequentially, the Fine-grained Visual Calibrating focuses on capturing the rhythmic dynamics of lip motion. Next, a Time-aware Context Aligning module integrates the textual features to ensure precise audio-visual synchronization. Finally, the designed JSAR mechanism regularizes the alignment stage through semantic and temporal consistency constraints, guiding the generative trajectory toward synchronized and high-fidelity speech.

![](images/0800f0ea8370f46e62920b6801b978b1107513124aca154132d1cd7a666cf301.jpg)  
Fig. 2: The overview of architecture

## 3.1 Acoustic and Semantic Prior

The reference audio $R _ { a }$ is extracted to a raw mel-spectrogram $m _ { r a w } \in \mathbb { R } ^ { F \times L }$ ， where F is the mel dimension and L is the sequence length. A binary temporal mask $\mathbf { M } _ { b } \in \{ 0 , 1 \} ^ { L }$ is then applied to obscure the target region η. This enables the model to yield the masked acoustic features $\mathcal { H } _ { m } = ( 1 - \mathbf { M } _ { b } ) \odot m _ { r a w }$ . Unlike directly concatenating the masked visual features, we construct a unified sequence by jointly modeling text and $\mathcal { H } _ { m }$ . To expand the word-level text sequence to the mel-level, we adopt two expansion strategies: (1) a padding operation to preserve the complete linguistic content; (2) a cross-attention operation to provide coarse temporal priors. Finally, we concatenate these textual sequences and $\mathcal { H } _ { m }$ into the semantic-acoustic prior, rather than using raw visual signals for early-stage conditioning.

## 3.2 CoSync-DiT

Traditional Diffusion Transformers (DiTs) struggle to meet the strict demands of movie dubbing. They fail to maintain precise audio-visual synchronization, degrading both timbre fidelity and pronunciation clarity. Drawing inspiration from the human dubbing cognitive process, we propose the Cognitive Synchronous Diffusion Transformer (CoSync-DiT). The model learns to progressively establish the reference speaker’s style, perform fine-grained lip calibration, and enforce contextual alignment.

Stage 1: Acoustic Style Adapting. In the scope of OT-CFM, our model predicts the vector field mapping from a standard Gaussian noise distribution $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ to the target speech latent $x _ { 1 }$ , following the time-dependent path $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ . In stage 1, the model concatenates the noisy speech $x _ { t }$ and the semantic-acoustic prior by unified projection layer to form the initial sequence $\mathcal { Z } ^ { 0 }$ . The Multi-Head Self-Attention (MHSA) models the long-range dependencies in the initial hidden states to capture style information:

$$
\mathcal { Z } _ { s t y l e } ^ { l } = \mathcal { Z } ^ { l - 1 } + \alpha _ { 1 } ^ { l } ( t ) \odot \mathrm { M H S A } ( \mathrm { A d a L N } _ { \beta _ { 1 } , \gamma _ { 1 } } ( \mathcal { Z } ^ { l - 1 } , t ) ) ,\tag{1}
$$

where l represents the current block level. $\operatorname { A d a L N } _ { \beta _ { 1 } , \gamma _ { 1 } }$ applies Time-Adaptive Layer Normalization (Time-AdaLN) to modulate the input feature statistics via scale $\gamma _ { 1 }$ and shift $\beta _ { 1 }$ vectors driven by the current time t. The gating term $\alpha _ { 1 } ( t )$ controls the residual contribution. Similarly, a time-adaptive Multilayer Perceptron (MLP) further refines these features via non-linear dimensional transformations to consolidate the relationship between acquired acoustic style and text.

Stage 2: Fine-grained Visual Calibrating. The input silent video is first processed by a lip-motion encoder to extract the raw lip features $\mathcal { X } _ { r a w }$ . Then, $\mathcal { X } _ { r a w }$ is then passed through a cascaded upsampling layer to yield the refined visual representations $\mathcal { X } _ { l i p }$ , ensuring their temporal resolution strictly matches that of the target mel-spectrogram. Next, $\mathcal { X } _ { l i p }$ are seamlessly injected into the $\mathcal { Z } _ { s t y l e } ^ { l }$ via a zero-initialized learnable gate:

$$
\mathcal { Z } _ { l i p } ^ { l } = \mathcal { Z } _ { s t y l e } ^ { l } + A ^ { l } \odot \mathcal { X } _ { l i p } , \quad \mathrm { w h e r e ~ \it A } ^ { l } = { \bf 0 } \in \mathbb { R } ^ { d } .\tag{2}
$$

This gating mechanism $\varLambda ^ { l }$ ensures that the visual injection acts as a subtle rhythmic residual, providing fine-grained frame-level calibration while protecting previously established style information.

Stage 3: Time-aware Context Aligning. We place the Multi-Head Cross-Attention (MHCA) at the bottom of the architecture to ensure that text alignment operates on fully mature visual-acoustic representations, rather than forcing a premature and unstable complementary fusion. Furthermore, we introduce a time-aware mechanism, AdaL $\mathrm { N } _ { \beta _ { 3 } , \gamma _ { 3 } }$ , to modulate the cross-attention process, allowing the alignment dynamics to better adapt to the evolving flow states:

$$
\mathcal { Z } _ { o u t } ^ { l } = \mathcal { Z } _ { l i p } ^ { l } + \alpha _ { 3 } ^ { l } ( t ) \odot ( \mathrm { M H C A } ( \mathrm { A d a L N } _ { \beta _ { 3 } , \gamma _ { 3 } } ( \mathcal { Z } _ { l i p } ^ { l } , t ) , \mathcal { H } _ { t e x t } ) ) ,\tag{3}
$$

where $\alpha _ { 3 } ^ { l } ( t )$ controls the residual contribution of the cross-attention output according to the current timestep t. The textual representation $\mathcal { H } _ { t e x t }$ extracted by ConvNeXtV2 encoder [42] serves as both the Key and Value, encouraging the generative path to converge toward the aligned content by implicitly retrieving the linguistic context.

## 3.3 Joint Semantic and Alignment Regularization

While the OT-CFM efficiently constructs the data generation path, the unconstrained vector field estimation often leads to temporal misalignments. In this work, we introduce the Joint Semantic and Alignment Regularization (JSAR) mechanism, constraining both the final flow hidden states and the intermediate context representations.

Alignment Regularization. The intermediate context representations $( i . e .$ the output of the MHCA in Eq. (3), denoted as $\hat { \mathcal { Z } } _ { c a } )$ need to maintain inherent temporal consistency when queried by the flow hidden features. To implicitly align these context representations in time, we introduce a frame-level contrastive learning by employing the audio branch representations from a pre-trained AV-HuBERT (denoted as $\mathcal { F } _ { a v } )$ as the contrastive ground truth. Both the outputs $\hat { \mathcal { Z } } _ { c a }$ and the AV-HuBERT features $\mathcal { F } _ { a v }$ are then L2-normalized along the feature dimension. Then, it uses the InfoNCE objective to maximize the cosine similarity of matching temporal frames while repelling non-matching frames:

$$
\mathcal { L } _ { C L } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp { \left( \langle \hat { z } _ { c a } ^ { ( i ) } , f _ { a v } ^ { ( i ) } \rangle / \tau \right) } } { \sum _ { j = 1 } ^ { N } \exp { \left( \langle \hat { z } _ { c a } ^ { ( i ) } , f _ { a v } ^ { ( j ) } \rangle / \tau \right) } } ,\tag{4}
$$

where $\hat { z } _ { c a } ^ { ( i ) }$ and $f _ { a v } ^ { ( i ) }$ represent the i-th frame of the flattened $\hat { \mathcal { Z } } _ { c a }$ and $\mathcal { F } _ { a v }$ respectively, ⟨·, ·⟩ denotes the dot product, and τ is the temperature hyperparameter.

Semantic Regularization. To guarantee pronunciation correctness, we apply a Connectionist Temporal Classification $\left( \mathrm { C T C } \right) \mathcal { L } _ { c t c }$ loss directly to the final hidden features $\mathcal { Z } _ { o u t } ^ { l } \left( i . e . \right.$ ., the output of Eq. (3)). It encourages the model to retain more linguistic information in predicted hidden states. By jointly optimizing these objectives within the JSAR framework, the model effectively synchronizes the context with the acoustic dynamics and maintains semantic consistency, stably anchoring the flow matching trajectory.

## 3.4 Optimal-Transport Conditional Flow Matching Objective

The objective of the OT-CFM is to minimize the Mean Squared Error (MSE) between the predicted vector field by CoSync-DiT and the target vector field:

$$
\mathcal { L } _ { f m } = \mathbb { E } _ { t , q ( x _ { 1 } ) , p ( x _ { 0 } ) } [ | | v _ { \theta } ( x _ { t } | t , \mathcal { H } _ { m } , \mathcal { X } _ { l i p } , \mathcal { H } _ { t e x t } ) - ( x _ { 1 } - x _ { 0 } ) | | ^ { 2 } ] ,\tag{5}
$$

where $t \sim \mathcal { U } [ 0 , 1 ]$ is the timestep, and $x _ { 1 } \ \sim \ q ( x _ { 1 } )$ denotes the target melspectrogram latent. $x _ { t }$ represents the intermediate noise latent. vθ represents the vector field predicted by the proposed CoSync-DiT with model parameters $\theta ,$ which is progressively conditioned on the $\mathcal { H } _ { m } , \mathcal { X } _ { l i p }$ , and $\mathcal { H } _ { t e x t }$

## 3.5 Acoustic-Semantic Classifier-Free Guidance

In conditional flow matching, classifier-free guidance (CFG) [16] is typically employed to amplify the influence of the conditioning signals. Driven by acoustic and semantic prior, our CFG focuses on explicitly decoupling these conditions. Let C denote the joint condition comprising both acoustic and semantic, and ∅ denote the unconditional prior. The modified vector field is formulated as:

$$
\begin{array} { r } { v _ { t , c f g } = v _ { t } ( x _ { t } , \mathcal { C } ) + \lambda _ { a } \cdot \big ( v _ { t } ( x _ { t } , \mathcal { C } ) - v _ { t } ( x _ { t } , \mathcal { H } _ { m } ) \big ) + \lambda _ { s } \cdot \big ( v _ { t } ( x _ { t } , \mathcal { H } _ { m } ) - v _ { t } ( x _ { t } , \mathcal { O } ) \big ) , } \end{array}\tag{6}
$$

where $\lambda _ { a }$ and $\lambda _ { s }$ are the acoustic and semantic guidance scales, enabling a highly controllable dubbing generation.

## 4 Experiments

## 4.1 Datasets and Experimental Setup

In this paper, we conduct experiments on three dubbing datasets, encompassing diverse in-the-wild scenarios $( e . g .$ , live action movies, vlogs, and dramas) as well as traditional scenarios. Tab. 1 summarizes the evaluation settings: (1) Setting 1: ground-truth speech is used as reference; (2) Setting 2: a different clip from the same speaker serves as reference; (3) Zero-shot: both unseen voice and unseen video are evaluated by out-of-domain dubbing dataset.

Table 1: Overview of evaluation configurations across three experimental settings in various dubbing benchmarks.
<table><tr><td>Dataset</td><td>Speaker</td><td>Main Scenes1</td><td>Experiment Setting</td></tr><tr><td>CelebV-Dub [38] Multi-SpeakersVlogs,Dramas</td><td></td><td></td><td>Setting1&amp; 2</td></tr><tr><td>Chem [17]</td><td></td><td>Single-SpeakerTeaching Video</td><td>Setting1&amp; 2</td></tr><tr><td></td><td>CinePile-Dub [35] Multi-Speakers Live Action Movies Zero-shot Setting</td><td></td><td></td></tr></table>

Chemistry Lecture (Chem). This single-speaker dataset [33] features a chemistry teacher delivering educational classroom lectures. It comprises short video segments sourced from YouTube and encompasses approximately nine hours of total video content. Following the standard sentence-level pre-processing [17], the dataset yields 6,132 training samples and 196 testing samples for dubbing.

CelebV-Dub. Built upon the foundations of CelebV-HQ [56] and CelebV-Text [48], this dataset aggregates content from diverse sources such as vlogs and dramas. It serves as a distinctly challenging benchmark by featuring unconstrained real-world settings alongside rich emotional variations. Following the official division [38], this dataset provides 79,933 training samples and 213 testing samples.

CinePile-Dub. Derived from the original CinePile [35] dataset, CinePile-Dub is specifically curated from professional live-action movie productions. It features high emotional intensity and expressive acting-specific prosody characteristic of authentic cinematic environments. Specifically, it comprises 160 professional movie video clips exclusively reserved to rigorously evaluate dubbing performance in a zero-shot setting.

## 4.2 Evaluation Metrics

We adopt several authority metrics to comprehensively evaluate various aspects of generated dubbing.

WER measures pronunciation clarity by calculating the word error rate derived from an automatic speech recognition model, whisper-large-V3 [34].

EMOSIM measures emotion similarity. We compute the cosine similarity between the synthesized speech and the ground truth by the Emotion2Vec model [29], which is a universal speech emotion representation model.

SPKSIM measures speaker similarity. In contrast to the prior metric in [24], i.e., speaker encoder cosine similarity (SECS) by GE2E model [41], we adopt the WavLM-TDNN model [4], which is widely adopted in speaker verification to ensure a reliable measure [10, 38].

Sync-KL also known as duration divergence [51]. Since visual-based alignment metrics (e.g., AVSync [6] and LSE [9]) still suffer from the complex visual changes and perturbations in in-the-wild scenarios, Sync-KL offers a more reliable manner by directly quantifying the duration discrepancy between GT and generated speech, leveraging the fact that GT speech is naturally synchronized with video. We will provide further comparisons of AV-Sync in the supplementary materials.

DNSMOS is used to objectively evaluate speech quality by approximating subjective human ratings (Mean Opinion Score, MOS). DNSMOS [36] (Deep Noise Suppression MOS) assesses speech clarity and background noise cleanliness.

MOS-N and MOS-S. We will provide human subjective rating results in supplementary materials, including MOS-N (Naturalness) and MOS-S (Similarity).

## 4.3 Implementation Details

The input and output dimensions of the unified projection layer are 712 and 1,024, respectively. ConvPosition is used to inject positional information into the unified projection layer with a kernel size of 31 and 16 groups. CoSync-DiT consists of 22 layers. The multi-head self-attention and multi-head crossattention layers both have a hidden dimension of 1,024 with 16 attention heads. Lip regions are resized to 96×96 and processed by a pre-trained AV-HuBERT [37] model to extract 1,024-dimensional embeddings. The text encoder comprises a stack of four ConvNeXt V2 blocks with a hidden dimension of 512. In JSAR, the temperature parameter τ is set to 0.07. The CTC projection layer within JSAR includes two temporal downsampling layers with Mish activations to map the 1,024-dimensional features into a 2,547-dimensional space. The final projection layer maps the 1,024-dimensional features to a predicted 100-dimensional vector field. During inference, the predicted vector field is used to solve the ODE from Gaussian noise to the target mel-spectrogram using an Euler solver with 32 function evaluations. We optimize the model using the AdamW optimizer [28] with momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The decoupled weight decay is set to 0.01, and the epsilon term is $1 \times 1 0 ^ { - 8 }$ . A random 70% to 100% span of mel-spectrogram frames is masked with span length η.

## 4.4 Performance Comparison with SOTA Methods

Results on the Chem (Setting 1). As shown in Tab. 2, our proposed CoSync-DiT achieves the best performance across all evaluated metrics. Under Setting 1, it attains 81.84% in SPKSIM, surpassing the second-best EmoDubber by an absolute margin of 6.24%, demonstrating exceptional capability in high-fidelity speaker preservation. Furthermore, our method achieves 87.84% in EMOSIM and the lowest WER of 7.04%, validating the effectiveness of our time-aware context aligning and semantic consistency constraints for precise pronunciation modeling. Finally, the lowest Sync-KL score of 0.289 and the highest DNSMOS of 3.83 among all dubbing baselines further confirm its superior fine-grained audio-visual synchronization and premium acoustic clarity.

Table 2: Compared with SOTA methods on the Chem benchmark under Setting 1.
<table><tr><td>Methods</td><td colspan="5">SPKSIM(%)↑WER(%)↓EMOSIM(%)↑ Sync-KL↓DNSMOS↑</td></tr><tr><td>GT</td><td>100.00</td><td>3.85</td><td>100.00</td><td>0.00</td><td>3.86</td></tr><tr><td>HPMDubbing g[11] (CVPR&#x27;23)</td><td>57.05</td><td>17.52</td><td>78.80</td><td>0.477</td><td>3.34</td></tr><tr><td>StyleDubber [13] (ACL&#x27;24)</td><td>61.87</td><td>10.62</td><td>80.95</td><td>0.440</td><td>3.39</td></tr><tr><td>EmoDubber [12] (CVPR&#x27;25)</td><td>75.60</td><td>11.86</td><td>85.10</td><td>0.420</td><td>3.70</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>75.37</td><td>11.51</td><td>83.47</td><td>0.417</td><td>3.66</td></tr><tr><td>HD-Dubber [24](TPAMI&#x27;25)</td><td>64.78</td><td>14.23</td><td>79.59</td><td>0.410</td><td>3.60</td></tr><tr><td>AlignDiT [6] (MM&#x27;25)</td><td>72.73</td><td>12.39</td><td>86.28</td><td>0.349</td><td>3.80</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>38.55</td><td>9.45</td><td>77.93</td><td>0.464</td><td>3.62</td></tr><tr><td>InstructDub [51] (AAAI26)</td><td>42.10</td><td>8.86</td><td>80.99</td><td>0.431</td><td>3.82</td></tr><tr><td>Ours</td><td>81.84</td><td>7.04</td><td>87.84</td><td>0.289</td><td>3.83</td></tr></table>

Table 3: Compared with SOTA methods on the Chem benchmark under Setting 2.
<table><tr><td colspan="2">Methods</td><td colspan="4">[SPKSIM(%)↑WER(%)↓EMOSIM(%)↑ Sync-KL↓DNSMOS↑</td></tr><tr><td>GT</td><td>73.01</td><td>3.85</td><td>100.00</td><td>0.00</td><td>3.86</td></tr><tr><td>HPMDubbing g[11] (CVPR23)</td><td>44.67</td><td>18.48</td><td>77.49</td><td>0.458</td><td>3.33</td></tr><tr><td>StyleDubber [13] (ACL&#x27;24)</td><td>50.00</td><td>11.98</td><td>79.09</td><td>0.426</td><td>3.34</td></tr><tr><td>EmoDubber r[12] (CVPR&#x27;25)</td><td>67.53</td><td>12.01</td><td>79.41</td><td>0.427</td><td>3.71</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>60.61</td><td>15.24</td><td>78.95</td><td>0.411</td><td>3.66</td></tr><tr><td>HD-Dubber [24](TPAMI25)</td><td>52.70</td><td>14.80</td><td>77.77</td><td>0.430</td><td>3.59</td></tr><tr><td>AlignDiT [6] (MM&#x27;25)</td><td>64.08</td><td>13.28</td><td>83.76</td><td>0.349</td><td>3.82</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>26.87</td><td>11.69</td><td>77.42</td><td>0.412</td><td>3.53</td></tr><tr><td>InstructDub [51] (AAAI26)</td><td>35.31</td><td>8.46</td><td>78.48</td><td>0.433</td><td>3.83</td></tr><tr><td>Ours</td><td>72.29</td><td>8.43</td><td>87.06</td><td>0.288</td><td>3.84</td></tr></table>

Results on the Chem (Setting 2). As presented in Tab. 3, we further evaluate the models under the distinctly more challenging Setting 2 scenario. Despite the increased difficulty, our proposed CoSync-DiT consistently maintains its superiority across all evaluated metrics. Specifically, it achieves 72.29% in SPKSIM, outperforming the second-best EmoDubber by an absolute margin of 4.76% and remarkably approaching the Ground Truth upper bound of 73.01%. This demonstrates the exceptional robustness of our acoustic style adapting phase in capturing complex unseen timbres. Furthermore, our method yields the highest EMOSIM of 87.06% and the lowest WER of 8.43%. Finally, CoSync-DiT attains the exceptionally low Sync-KL score of 0.288 and a remarkable DNSMOS of 3.84, confirming that our fine-grained visual calibrating module effectively guarantees strict audio-visual synchronization without sacrificing premium acoustic fidelity.

Results on the CelebV-Dub (Setting 1). Results are reported in Tab. 4. Compared with relatively controlled datasets like Chem, CelebV-Dub contains diverse in-the-wild content such as vlogs and dramas with rich emotional variations, posing a distinctly more challenging evaluation. Despite these complexities, our proposed CoSync-DiT successfully achieves state-of-the-art performance across all evaluated metrics. Specifically, it attains 65.21% in SPKSIM, outperforming the strongest baseline AlignDiT by an absolute margin of 5.50%, which confirms the reliability of our acoustic style adapting phase in handling highly expressive speech. Furthermore, our method obtains the highest EMOSIM of 84.61% alongside an exceptionally low WER of 4.29%, closely approaching the Ground Truth limit of 4.15%. Finally, securing the lowest Sync-KL score of 0.392 and the highest DNSMOS of 3.46 proves that our framework consistently delivers precise audio-visual synchronization and premium acoustic fidelity even in challenging dubbing scenarios.

Table 4: Compared with SOTA Dubbing methods on the CelebV-Dub dataset under Setting 1. Unlike single dubbing scenario (e.g., Chem), CelebV-Dub includes diverse video content (vlogs and dramas) with highly expressive speech.
<table><tr><td>Methods</td><td colspan="5">SPKSIM(%)↑WER(%)↓EMOSIM(%)↑ Sync-KL↓ DNSMOS↑</td></tr><tr><td>GT</td><td>100.00</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td></tr><tr><td>HPMDubbing [11] (CVPR&#x27;23)</td><td>17.26</td><td>21.99</td><td>73.01</td><td>0.441</td><td>2.47</td></tr><tr><td>StyleDubber [13](ACL&#x27;24)</td><td>26.39</td><td>10.79</td><td>79.38</td><td>0.415</td><td>2.65</td></tr><tr><td>EmoDubber [12] (CVPR&#x27;25)</td><td>22.08</td><td>13.44</td><td>76.59</td><td>0.407</td><td>3.26</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>22.24</td><td>13.95</td><td>76.76</td><td>0.423</td><td>3.27</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>26.97</td><td>5.18</td><td>73.49</td><td>0.421</td><td>3.12</td></tr><tr><td>HD-Dubber [24] (TPAMI&#x27;25)</td><td>10.96</td><td>66.30</td><td>19.73</td><td>0.410</td><td>3.01</td></tr><tr><td>AlignDiT [6] (MM&#x27;25)</td><td>59.71</td><td>9.48</td><td>84.54</td><td>0.402</td><td>3.45</td></tr><tr><td>InstructDub [51] (AAAI26)</td><td>29.12</td><td>6.13</td><td>75.57</td><td>0.429</td><td>3.16</td></tr><tr><td>Ours</td><td>65.21</td><td>4.29</td><td>84.61</td><td>0.392</td><td>3.46</td></tr></table>

Table 5: Compared with SOTA methods on the CelebV-Dub dataset under Setting 2.
<table><tr><td>Methods</td><td colspan="5">SPKSIM(%)↑WER(%)↓EMOSIM(%)↑ Sync-KL↓DNSMOS↑</td></tr><tr><td>GT</td><td>66.13</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td></tr><tr><td>HPMDubbing [11] (CVPR&#x27;23)</td><td>12.52</td><td>23.64</td><td>69.25</td><td>0.447</td><td>2.47</td></tr><tr><td>StyleDubber [13] (ACL&#x27;24)</td><td>19.42</td><td>7.03</td><td>74.87</td><td>0.405</td><td>2.63</td></tr><tr><td>EmoDubber [12] (CVPR&#x27;25)</td><td>18.78</td><td>16.37</td><td>76.30</td><td>0.422</td><td>3.30</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>19.32</td><td>14.94</td><td>74.18</td><td>0.420</td><td>3.31</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>23.13</td><td>6.41</td><td>71.38</td><td>0.432</td><td>3.14</td></tr><tr><td>HD-Dubber [24](TPAM&#x27;25)</td><td>9.69</td><td>64.71</td><td>16.86</td><td>0.400</td><td>3.00</td></tr><tr><td>AlignDiT [6](MM&#x27;25)</td><td>49.49</td><td>13.18</td><td>79.69</td><td>0.413</td><td>3.47</td></tr><tr><td>InstructDub [51] (AAAI&#x27;26)</td><td>22.85</td><td>5.64</td><td>74.03</td><td>0.434</td><td>3.18</td></tr><tr><td>Ours</td><td>53.44</td><td>6.39</td><td>80.29</td><td>0.381</td><td>3.47</td></tr></table>

Results on the CelebV-Dub (Setting 2). Finally, we evaluate the models on the CelebV-Dub dataset under Setting 2. As shown in Tab. 5, our CoSync-DiT achieves the highest SPKSIM of 53.44% and EMOSIM of 80.29%, demonstrating robust acoustic style adapting capabilities for complex unseen voices. While InstructDub attains a marginally lower Word Error Rate of 5.64%, it severely compromises speaker identity preservation by yielding a remarkably low SPKSIM of 22.85%. In contrast, our method maintains a highly competitive WER of 6.39% while simultaneously establishing the best audio-visual synchronization with a Sync-KL of 0.381. Furthermore, achieving the highest DNSMOS score of 3.47 confirms that our dubbing method avoids the pitfall of single-metric overoptimization, providing the most comprehensive and balanced dubbing quality.

Table 6: Zero-shot main experiment results on CinePile-Dub dataset (Out-of-Domain). All models are trained on the training set of the basic dubbing dataset CelebV-Dub.
<table><tr><td>Methods</td><td colspan="5">[SPKSIM(%)↑ WER(%)↓ EMOSIM(%)↑ Sync-KL↓ DNSMOS↑</td></tr><tr><td>GT</td><td>100.00</td><td>2.56</td><td>100.00</td><td>0.00</td><td>3.15</td></tr><tr><td>HPMDubbing [11](CVPR&#x27;23)</td><td>42.72</td><td>99.20</td><td>61.22</td><td>0.391</td><td>3.23</td></tr><tr><td>StyleDubber [13](ACL&#x27;24)</td><td>41.65</td><td>74.96</td><td>63.88</td><td>0.362</td><td>2.89</td></tr><tr><td>HD-Dubber [24]](TPAMI&#x27;25)</td><td>5.84</td><td>42.75</td><td>56.32</td><td>0.337</td><td>2.58</td></tr><tr><td>EmoDubber [12](CVPR&#x27;25)</td><td>23.10</td><td>49.57</td><td>70.08</td><td>0.337</td><td>3.05</td></tr><tr><td>FlowDubber [10](MM&#x27;25)</td><td>22.68</td><td>54.41</td><td>67.39</td><td>0.354</td><td>3.04</td></tr><tr><td>AlignDiT [6](MM&#x27;25)</td><td>58.90</td><td>20.98</td><td>77.39</td><td>0.342</td><td>3.36</td></tr><tr><td>ProDubber [52](CVPR&#x27;25)</td><td>28.86</td><td>5.25</td><td>70.35</td><td>0.335</td><td>2.93</td></tr><tr><td>InstructDub [51](AAAI26)</td><td>27.61</td><td>4.61</td><td>65.97</td><td>0.370</td><td>2.95</td></tr><tr><td>Ours</td><td>60.04</td><td>5.59</td><td>77.41</td><td>0.332</td><td>3.40</td></tr></table>

![](images/e509f589150b4adaf1291e5df713e5997455bbe4d85fc4e4809581a3fa5e3646.jpg)

![](images/037fbd0b2361dce28f5ac6aa14e1aeffb1c9020ef54e9877eb296b7450cb78b4.jpg)  
Fig. 3: Performance comparison between the proposed method and the SOTA baseline under the different Number of Function Evaluations (NFE) steps.

Results on the CinePile-Dub (Zero-shot Setting). To assess true zero-shot generalization, we conduct an out-of-domain evaluation where models trained exclusively on CelebV-Dub are tested on CinePile-Dub, a highly challenging realworld live-action movie dataset. As presented in Tab. 6, our proposed CoSync-DiT achieves the highest SPKSIM of 60.04% and EMOSIM of 77.41%. This confirms the powerful cross-domain extrapolation capacity of the proposed method when confronted with entirely novel cinematic environments. Although Instruct-Dub attains the lowest Word Error Rate of 4.61%, it experiences a catastrophic collapse in speaker identity preservation by yielding a poor SPKSIM of 27.61%. Conversely, our method maintains a highly competitive WER of 5.59% while simultaneously establishing the best speaker similarity 60.04% and audio-visual synchronization with a leading Sync-KL score of 0.332.

Table 7: Results of ablation study on CelebV-Dub dataset under Setting 2.
<table><tr><td>Methods</td><td colspan="4">|SPKSIM(%)↑WER（%)↓EMOSIM(%)↑Sync-KL↓DNSMOS↑</td></tr><tr><td>w/o Style Adapting</td><td>19.64 6.84</td><td>77.24</td><td>0.385</td><td>3.38</td></tr><tr><td>w/o Visual Calibrating</td><td>53.25 6.40</td><td>80.17</td><td>0.419</td><td>3.45</td></tr><tr><td>w/o Context Aligning</td><td>52.75 7.39</td><td>80.04</td><td>0.446</td><td>3.44</td></tr><tr><td>w/o JSAR</td><td>51.30 8.72</td><td>80.14</td><td>0.431</td><td>3.39</td></tr><tr><td>w/o Semantic Consistency</td><td>52.34 8.39</td><td>80.19</td><td>0.392</td><td>3.42</td></tr><tr><td>w/o Temporal Consistency</td><td>51.37 6.58</td><td>80.24</td><td>0.425</td><td>3.44</td></tr><tr><td>Full model</td><td>53.44 6.39</td><td>80.29</td><td>0.381</td><td>3.47</td></tr></table>

## 4.5 Ablation Study

To validate the effectiveness of our core designs, we conduct a comprehensive ablation study on the CelebV-Dub dataset under Setting 2. As reported in Tab. 7, removing the acoustic style adapting phase triggers a severe collapse in speaker similarity, plummeting the SPKSIM from 53.44% to 19.64%. This confirms its indispensable role in capturing and preserving the target timbre. When the fine-grained visual calibrating module is omitted, the Sync-KL metric clearly degrades to 0.419, indicating its absolute necessity for establishing strict lip-sync. Furthermore, we find that the absence of the time-aware context aligning phase exerts the most profound impact on overall synchronization, as its core function relies on re-retrieving textual content conditioned on previously integrated information to adjust articulation. Finally, we dissect the JSAR mechanism. Eliminating the semantic consistency constraint primarily damages the pronunciation accuracy, dropping the temporal consistency constraint strictly deteriorates the audio-visual synchronization, and discarding the entire JSAR framework compounds both errors.

## 4.6 Comprehensive Analysis of Generative Robustness

To comprehensively evaluate the inference efficiency and generation stability of our approach against the state-of-the-art baseline, we conduct a comparative analysis under various Number of Function Evaluations (NFE) settings. As illustrated in Fig. 3, AlignDiT suffers a catastrophic performance collapse at extremely low sampling steps, evidenced by its SIM-O metric plummeting to approximately 0.30 at an NFE of 8. In stark contrast, our method exhibits remarkable robustness by maintaining a consistently high SIM-O of over 0.65 across all evaluated NFE configurations. Furthermore, CoSync-DiT continuously secures a significantly lower Word Error Rate than AlignDiT at every corresponding step, achieving its optimal pronunciation clarity around an NFE of 16 to 32. These results compellingly demonstrate that our method establishes a robust dubbing system, unlocking highly efficient few-step inference without sacrificing acoustic or synchronization quality.

![](images/b1e0f54b1ce1b5db4ce88f40bf2cd4481e7b019a977388b0c128a80f13c8739a.jpg)  
Fig. 4: Visual comparison of the generated mel-spectrograms. The blue arrows highlight specific regions requiring attention for audio-visual synchronization.

## 4.7 Qualitative Comparison

We visualize the mel-spectrograms of the ground truth and the generated speech from different models in Fig. 4 (additional results are provided in the Appendix). The blue dashed regions and arrows highlight critical temporal intervals for audio-visual synchronization, while the white dashed boxes emphasize fine-grained pronunciation differences. While most baseline methods perform adequately on the controlled Chem dataset, they degrade sharply on more challenging benchmarks. Specifically in vlog and cinematic scenarios, models like InstructDubber and ProDubber completely fail to maintain proper audio-visual sync. Furthermore, AlignDiT exhibits noticeable temporal shifts and little distortions within the arrow-indicated regions. Conversely, our proposed CoSync-DiT synthesizes highly robust mel-spectrograms that most closely match the ground truth across both boundaries and details.

## 5 Conclusion

In this paper, we propose CoSync-DiT, a flow matching-based framework for movie dubbing, which sequentially executes acoustic style adapting, fine-grained visual calibrating, and time-aware context aligning to explicitly guide the noiseto-speech generative trajectory. Furthermore, we introduce the Joint Semantic and Alignment Regularization (JSAR) mechanism to simultaneously enforce frame-level temporal consistency on the contextual outputs and semantic consistency on the flow hidden states, guaranteeing reliable pronunciation during the alignment phase. Extensive experiments on standard benchmarks and in-the-wild datasets demonstrate that our method achieves state-of-the-art performance.

## Appendix

This appendix provides the following extra content:

– Appendix A provides additional quantitative results using the AVSync metric.

– Appendix B presents experimental results under an extreme configuration combining Zero-shot and Setting 2.

– Appendix C includes comparisons with the official checkpoint of AlignDiT trained on the large-scale LRS3 dataset.

– Appendix D provides further qualitative visualizations and mel-spectrogram analysis.

We will open-source all detailed experimental settings, source code, and pretrained weights.

## A Synchronization Analysis (AVSync)

In this section, we expand upon our primary experimental results by introducing the AVSync metric to comprehensively evaluate the effectiveness of our proposed method. Similar to Sync-KL [51], AVSync is a recently proposed metric specifically designed to assess fine-grained lip synchronization [6]. As highlighted in recent work [6,43], AVSync provides a substantially more accurate evaluation of audio-visual alignment than traditional metrics such as LSE-C and LSE-D [9]. Mechanistically, AVSync evaluates this synchronization by computing the cosine similarity between the AV-HuBERT [37] features extracted from the video paired with ground-truth speech and those extracted from the video paired with the synthesized speech.

Table 8: Compared with SOTA Dubbing methods on the CelebV-Dub dataset under Setting 1. Unlike single dubbing scenario (e.g., Chem), CelebV-Dub includes diverse video content (vlogs and dramas) with highly expressive speech.
<table><tr><td>Methods</td><td colspan="6">[SPKSIM(%)↑WER(%)↓EMOSIM(%)↑ Sync-KL↓DNSMOS↑AVSync(%)↑</td></tr><tr><td>GT</td><td>100.00</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td><td>100.00</td></tr><tr><td>HPMDubbing [11] (CVPR&#x27;23)</td><td>17.26</td><td>21.99</td><td>73.01</td><td>0.441</td><td>2.47</td><td>47.49</td></tr><tr><td>StyleDubber [13](ACL&#x27;24)</td><td>26.39</td><td>10.79</td><td>79.38</td><td>0.415</td><td>2.65</td><td>21.38</td></tr><tr><td>EmoDubber [12] (CVPR&#x27;25)</td><td>22.08</td><td>13.44</td><td>76.59</td><td>0.407</td><td>3.26</td><td>32.94</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>22.24</td><td>13.95</td><td>76.76</td><td>0.423</td><td>3.27</td><td>33.40</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>26.97</td><td>5.18</td><td>73.49</td><td>0.421</td><td>3.12</td><td>15.96</td></tr><tr><td>HD-Dubber [24] (TPAMI&#x27;25)</td><td>10.96</td><td>66.30</td><td>19.73</td><td>0.410</td><td>3.01</td><td>18.83</td></tr><tr><td>AlignDiT [6] (MM&#x27;25)</td><td>59.71</td><td>9.48</td><td>84.54</td><td>0.402</td><td>3.45</td><td>49.05</td></tr><tr><td>InstructDub[ [51] (AAAI26)</td><td>29.12</td><td>6.13</td><td>75.57</td><td>0.429</td><td>3.16</td><td>16.74</td></tr><tr><td>Ours</td><td>65.21</td><td>4.29</td><td>84.61</td><td>0.392</td><td>3.46</td><td>65.94</td></tr></table>

Table 9: Compared with SOTA methods on the CelebV-Dub dataset under Setting 2.
<table><tr><td>Methods</td><td colspan="6">[SPKSIM(%)↑WER(%)↓ EMOSIM(%)↑ Sync-KL↓DNSMOS↑ AVSync(%)↑</td></tr><tr><td>GT</td><td>66.13</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td><td>100.00</td></tr><tr><td>HPMDubbing [11] (CVPR23)</td><td>12.52</td><td>23.64</td><td>69.25</td><td>0.447</td><td>2.47</td><td>42.67</td></tr><tr><td>StyleDubber [13] (ACL&#x27;24)</td><td>19.42</td><td>7.03</td><td>74.87</td><td>0.405</td><td>2.63</td><td>20.75</td></tr><tr><td>EmoDubber [12] (CVPR&#x27;25)</td><td>18.78</td><td>16.37</td><td>76.30</td><td>0.422</td><td>3.30</td><td>32.14</td></tr><tr><td>FlowDubber [10] (MM&#x27;25)</td><td>19.32</td><td>14.94</td><td>74.18</td><td>0.420</td><td>3.31</td><td>32.37</td></tr><tr><td>Produbber [52] (CVPR&#x27;25)</td><td>23.13</td><td>6.41</td><td>71.38</td><td>0.432</td><td>3.14</td><td>14.92</td></tr><tr><td>HD-Dubber [24](TPAMI25)</td><td>9.69</td><td>64.71</td><td>16.86</td><td>0.400</td><td>3.00</td><td>18.45</td></tr><tr><td>AlignDiT [6] (MM&#x27;25)</td><td>49.49</td><td>13.18</td><td>79.69</td><td>0.413</td><td>3.47</td><td>48.72</td></tr><tr><td>InstructDub [51] (AAAI26)</td><td>22.85</td><td>5.64</td><td>74.03</td><td>0.434</td><td>3.18</td><td>16.16</td></tr><tr><td>Ours</td><td>53.44</td><td>6.39</td><td>80.29</td><td>0.381</td><td>3.47</td><td>52.12</td></tr></table>

As reported in Tab. 8, our method consistently achieves state-of-the-art results across all metrics after incorporating the AVSync evaluation. Specifically, our approach obtains an AVSync score of 65.94%, which outperforms the strongest baseline AlignDiT by an absolute margin of 16.89%. This confirms the effectiveness of our proposed cognitive synchronous diffusion mechanism. By progressively guiding the denoising process, our method introduces acoustic style adapting, fine-grained visual calibrating, and time-aware context aligning at different network stages. This structured integration effectively prevents feature entanglement during the early sampling steps and establishes a precise temporal alignment between the generated speech and the target lip movements. Furthermore, under the challenging Setting 2 scenario of CelebV-Dub (see Tab. 9), although our method marginally trails the powerful TTS-pretrained baseline InstructDubber in Word Error Rate, it decisively surpasses all comparative methods in both speaker similarity and comprehensive synchronization metrics (both Sync-KL and AVSync). Consistent with these findings, our method sustains this comprehensive superiority in the out-of-domain movie evaluation (see Tab. 10), confirming its robust generalization capabilities and optimal alignment performance across unseen domains.

Table 10: Zero-shot results on CinePile-Dub dataset (Out-of-Domain Movie). All models are trained on the training set of the basic dubbing dataset CelebV-Dub.
<table><tr><td>Methods</td><td colspan="6">ISPKSIM(%)↑ WER(%)↓ EMOSIM(%)↑ Sync-KL↓DNSMOS↑ AVSync(%)↑</td></tr><tr><td>GT</td><td>100.00</td><td>2.56</td><td>100.00</td><td>0.00</td><td>3.15</td><td>100.00</td></tr><tr><td>HPMDubbing [11](CVPR23)</td><td>42.72</td><td>99.20</td><td>61.22</td><td>0.391</td><td>3.23</td><td>17.00</td></tr><tr><td>StyleDubber [13](ACL&#x27;24)</td><td>41.65</td><td>74.96</td><td>63.88</td><td>0.362</td><td>2.89</td><td>22.56</td></tr><tr><td>HD-Dubber [24]](TPAMI&#x27;25)</td><td>5.84</td><td>42.75</td><td>56.32</td><td>0.337</td><td>2.58</td><td>19.33</td></tr><tr><td>EmoDubber [12](CVPR&#x27;25)</td><td>23.10</td><td>49.57</td><td>70.08</td><td>0.337</td><td>3.05</td><td>13.75</td></tr><tr><td>FlowDubber [10](MM&#x27;25)</td><td>22.68</td><td>54.41</td><td>67.39</td><td>0.354</td><td>3.04</td><td>13.58</td></tr><tr><td>AlignDiT [6](MM&#x27;25)</td><td>58.90</td><td>20.98</td><td>77.39</td><td>0.342</td><td>3.36</td><td>31.77</td></tr><tr><td>ProDubber [52](CVPR&#x27;25)</td><td>28.86</td><td>5.25</td><td>70.35</td><td>0.335</td><td>2.93</td><td>21.79</td></tr><tr><td>InstructDub [5i](AAAI&#x27;26)</td><td>27.61</td><td>4.61</td><td>65.97</td><td>0.370</td><td>2.95</td><td>17.65</td></tr><tr><td>Ours</td><td>60.04</td><td>5.59</td><td>77.41</td><td>0.332</td><td>3.40</td><td>45.24</td></tr></table>

## B Robustness Analysis in Challenging Scenarios (both Zero-Shot and Setting 2)

Although existing experimental configurations are sufficiently challenging, they primarily cover CelebV-Dub under Setting 1 and Setting 2 alongside a standard zero-shot evaluation on out-of-domain movies. In this section, we introduce a significantly more rigorous setting that integrates Setting 2 into the zero-shot evaluation. This setup simultaneously removes both the training domain priors and the interference of target speech, ultimately creating a highly demanding evaluation scenario. We systematically summarize all configurations in Tab. 11. We hope this robust evaluation design promotes the advancement of the reliable visual dubbing field and communities. We plan to open-source all detailed experimental settings, source code, and pre-trained weights.

Table 11: Summary of the experiment configurations in the supplementary materials. Note that Setting 1 and Setting 2 define the reference audio selection strategy. Setting 1 uses the target speech as the reference audio, while Setting 2 uses an unaligned utterance from the same speaker. The Zero-shot condition indicates whether the target domain data is excluded during the training phase.
<table><tr><td>Table</td><td></td><td>|Reference audio Setting|Zero-shot Setting (OOD Movie)</td></tr><tr><td>Tab.8</td><td>Setting 1</td><td>×</td></tr><tr><td>Tab.9</td><td>Setting 2</td><td>×</td></tr><tr><td>Tab.10l</td><td>Setting 1</td><td>√</td></tr><tr><td>Tab. 12</td><td>Setting 2</td><td>√</td></tr><tr><td>Tab. 13</td><td>Seting 1</td><td>√</td></tr><tr><td>Tab. 14</td><td>Setting 2</td><td>√</td></tr></table>

As reported in Tab. 12, the performance of all evaluated methods experiences a noticeable decline under this highly constrained configuration. Despite these extreme challenges, our proposed method consistently secures the best performance across all objective metrics. Specifically, our approach achieves the highest speaker similarity of 47.24% and emotional similarity of 70.12%. This demonstrates its superior capability to reliably extract and transfer target vocal characteristics from temporally unaligned reference segments. Furthermore, our model attains the lowest Word Error Rate of 7.53%, successfully surpassing the strong InstructDubber baseline. Regarding synchronization, our method records the most precise Sync-KL of 0.319 and an AVSync score of 31.79%, outperforming the strongest baseline AlignDiT by a solid absolute margin of 9.29%. These comprehensive results compellingly confirm the exceptional robustness of our CoSync-DiT. It proves that our progressive cognitive synchronous architecture can reliably maintain high-fidelity speech generation and strict audio-visual alignment even in the most demanding out-of-domain scenarios.

Table 12: Zero-shot results on the CinePile-Dub dataset (Out-of-Domain Movie) under Setting 2. All models are trained on the training set of the basic dubbing dataset CelebV-Dub. The reference audio is from different segments of the same speaker.
<table><tr><td>Methods</td><td colspan="6">ISPKSIM(%)↑WER(%)↓ EMOSIM(%)↑ Sync-KL↓DNSMOS↑ AVSync(%)↑</td></tr><tr><td>GT</td><td>60.58</td><td>2.56</td><td>100.00</td><td>0.00</td><td>3.15</td><td>100.00</td></tr><tr><td>HPMDubbing [11](CVPR&#x27;23)</td><td>30.75</td><td>93.86</td><td>52.93</td><td>0.382</td><td>3.24</td><td>13.87</td></tr><tr><td>StyleDubber [13](ACL&#x27;24)</td><td>28.67</td><td>77.61</td><td>57.15</td><td>0.372</td><td>2.88</td><td>16.48</td></tr><tr><td>HD-Dubber [24](TPAMI25)</td><td>7.04</td><td>64.12</td><td>47.66</td><td>0.330</td><td>2.58</td><td>18.86</td></tr><tr><td>EmoDubber [12](CVPR&#x27;25)</td><td>19.43</td><td>57.73</td><td>62.46</td><td>0.347</td><td>3.05</td><td>13.00</td></tr><tr><td>FlowDubber [10](MM&#x27;25)</td><td>19.86</td><td>55.93</td><td>63.86</td><td>0.346</td><td>3.05</td><td>13.05</td></tr><tr><td>AlignDiT [6](MM&#x27;25)</td><td>38.58</td><td>34.52</td><td>66.74</td><td>0.358</td><td>3.37</td><td>22.50</td></tr><tr><td>ProDubber [52](CVPR&#x27;25)</td><td>23.91</td><td>8.02</td><td>62.06</td><td>0.338</td><td>2.93</td><td>20.80</td></tr><tr><td>InstructDub [51](AAAI&#x27;26)</td><td>23.11</td><td>7.62</td><td>61.14</td><td>0.367</td><td>2.94</td><td>16.66</td></tr><tr><td>Ours</td><td>47.24</td><td>7.53</td><td>70.12</td><td>0.319</td><td>3.39</td><td>31.79</td></tr></table>

## C Comparison with the Official AlignDiT Checkpoint

To the best of our knowledge, AlignDiT [6] represents one of the most advanced methods in the video-to-speech field. The official AlignDiT model is trained on the massive LRS3 dataset [1], which comprises approximately 131,000 utterances and 439 hours of unconstrained video content. In our primary experiments, we trained both our proposed method and the AlignDiT baseline on the CelebV-Dub dataset to ensure fairness. CelebV-Dub contains 67,549 samples totaling around 85.94 hours. Note that CelebV-Dub is currently the most popular and accessible large-scale dubbing dataset. Unfortunately, the complete LRS3 dataset is no longer available for public download from its official source. This restriction physically prevents us from training our model directly on the LRS3 dataset. Nevertheless, to conduct a comprehensive and fair comparison with the most powerful AlignDiT variant, we introduce an extended evaluation in this section. Specifically, we utilize the official pre-trained weights released by the AlignDiT authors to guarantee a rigorous and unbiased assessment. We denote this official LRS3-trained model as AlignDiT\*. To establish a strictly neutral testing ground, we evaluate all models on the out-of-domain CinePile-Dub cinematic dataset. This configuration ensures a completely fair zero-shot scenario for all competitors.

The comprehensive zero-shot quantitative results are reported in Tab. 13 (Zero-shot & Setting1) and Tab. 14 (Zero-shot & Setting2). First, we observe that the AlignDiT model trained on CelebV-Dub achieves performance highly comparable to the official AlignDiT\* model trained on LRS3. This directly confirms that our baseline implementation on CelebV-Dub is solid and does not disadvantage the comparative architecture. More importantly, our proposed method consistently outperforms both AlignDiT variants across all objective metrics. Even though our model is trained on a dataset approximately one-fifth the size of LRS3, it significantly suppresses the Word Error Rate and achieves much higher AVSync scores in both settings. This compellingly demonstrates that our architectural design possesses superior cross-modal alignment capabilities and stronger zero-shot generalization than merely scaling up the training data volume.

Table 13: Zero-shot evaluation results on the CinePile-Dub dataset (Out-of-domain Movie) under Setting 1. The asterisk (\*) denotes the official AlignDiT checkpoint pretrained on the large-scale LRS3 dataset. All evaluated models face completely unseen target domains.
<table><tr><td>Methods</td><td colspan="6">[SPKSIM(%)↑WER(%)↓] EMOSIM(%)↑ Sync-KL↓ DNSMOS↑ AVSync(%)↑</td></tr><tr><td>GT</td><td>100.00</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td><td>100.00</td></tr><tr><td>AlignDiT* [6]]</td><td>61.51</td><td>18.35</td><td>76.33</td><td>0.338</td><td>3.25</td><td>24.03</td></tr><tr><td>AlignDiT[6]</td><td>58.90</td><td>20.98</td><td>77.39</td><td>0.342</td><td>3.36</td><td>31.77</td></tr><tr><td>Ours</td><td>60.04</td><td>5.59</td><td>77.41</td><td>0.332</td><td>3.40</td><td>45.24</td></tr></table>

Table 14: Zero-shot evaluation results on the CinePile-Dub dataset (Out-of-domain Movie) under Setting 2. All evaluated models face completely unseen target domains and non-target reference audio. The asterisk (\*) denotes the official AlignDiT checkpoint pre-trained on the large-scale LRS3 dataset.
<table><tr><td>Methods</td><td colspan="6">[SPKSIM(%)↑WER(%)↓1 EMOSIM(%)↑ Sync-KL↓ DNSMOS↑ AVSync(%)↑</td></tr><tr><td>GT</td><td>100.00</td><td>4.15</td><td>100.00</td><td>0.000</td><td>3.38</td><td>100.00</td></tr><tr><td>AlignDiT* [6]]</td><td>44.01</td><td>35.90</td><td>68.80</td><td>0.359</td><td>3.23</td><td>18.72</td></tr><tr><td>AlignDiT[6]</td><td>38.58</td><td>34.52</td><td>66.74</td><td>0.358</td><td>3.37</td><td>22.50</td></tr><tr><td>Ours</td><td>47.24</td><td>7.53</td><td>70.12</td><td>0.319</td><td>3.39</td><td>31.79</td></tr></table>

![](images/51caf4f9755ae3a9e598e880a42ac1166e32adc0d704bf6d7443584f0db0d484.jpg)  
Fig. 5: Visual comparison of the generated mel-spectrograms with ground truth. The blue and white bounding boxes highlight regions where different models exhibit significant differences in duration and spectrogram details. Furthermore, the blue arrows pinpoint specific temporal regions that require critical attention for evaluating audiovisual synchronization.

![](images/cd0d1045be8647fe92ede8ab1c59ad3f046a0f5a02c48913919a237c413a0e2e.jpg)  
Fig. 6: Visual comparison of the generated mel-spectrograms with ground truth under zero-shot setting (out-of-domain movie scenario). The blue and white bounding boxes highlight regions where different models exhibit significant differences in duration and spectrogram details. Furthermore, the blue arrows pinpoint specific temporal regions that require critical attention for evaluating audio-visual synchronization.

## D More Qualitative Comparison

In this section, we provide additional visual examples of the generated melspectrograms to further evaluate the generated quality. As illustrated in Fig. 5 and Fig. 6, our proposed method demonstrates a distinct advantage in preserving both fine-grained acoustic details and audio-visual alignment. We specifically highlight the critical temporal regions indicated by the blue arrows. For instance, in the middle column of Fig. 5, this baseline exhibits a slight temporal shift. Similarly, in the middle column of Fig. 6, the state-of-the-art AlignDiT baseline completely misses the specific acoustic regions that should be synchronized with the ground truth. Conversely, our method, shown in the second row, reconstructs these specific acoustic patterns with strict temporal alignment. Furthermore, by observing the white rectangular boxes, our method presents highly clear spectrogram details. This directly reflects the robust timbre preservation and pronunciation clarity of our generated speech. This visual evidence intuitively validates the effectiveness of our method in guaranteeing precise audio-visual synchronization. Additionally, comparing our approach with duration predictor-based methods like InstructDubber and ProDubber reveals a fundamental architectural limitation of those baselines. Although such methods yield high pronunciation clarity, they completely fail to preserve audio-visual alignment in practical dubbing scenarios. As indicated by the blue rectangular boxes, their temporal synchronization is highly inaccurate.

## References

1. Afouras, T., Chung, J.S., Zisserman, A.: LRS3-TED: a large-scale dataset for visual speech recognition. CoRR abs/1809.00496 (2018)

2. Alburger, J.R.: The art of voice acting: The craft and business of performing for voiceover. Focal Press (2023)

3. Chen, Q., Tan, M., Qi, Y., Zhou, J., Li, Y., Wu, Q.: V2C: visual voice cloning. In: CVPR. pp. 21210–21219 (2022)

4. Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., Yu, X., Wei, F.: Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE J. Sel. Top. Signal Process. 16(6), 1505–1518 (2022)

5. Chen, Y., Niu, Z., Ma, Z., Deng, K., Wang, C., Zhao, J., Yu, K., Chen, X.: F5- tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885 (2024)

6. Choi, J., Kim, J.H., Sung-Bin, K., Oh, T.H., Chung, J.S.: Aligndit: Multimodal aligned diffusion transformer for synchronized speech generation. In: ACM MM. p. 10758–10767 (2025)

7. Choi, J., Kim, M., Ro, Y.M.: Intelligible lip-to-speech synthesis with speech units. In: Annu. Conf. Int. Speech Commun. Assoc. pp. 4349–4353 (2023)

8. Choi, J., Park, S.J., Kim, M., Ro, Y.M.: Av2av: Direct audio-visual speech to audio-visual speech translation with unified audio-visual speech representation. In: CVPR. pp. 27325–27337 (2024)

9. Chung, J.S., Zisserman, A.: Out of time: Automated lip sync in the wild. In: ACCV 2016 Workshops. pp. 251–263 (2016)

10. Cong, G., Li, L., Pan, J., Zhang, Z., Beheshti, A., van den Hengel, A., Qi, Y., Huang, Q.: Flowdubber: Movie dubbing with llm-based semantic-aware learning and flow matching based voice enhancing. In: ACM MM. p. 905–914 (2025)

11. Cong, G., Li, L., Qi, Y., Zha, Z.J., Wu, Q., Wang, W., Jiang, B., Yang, M.H., Huang, Q.: Learning to dub movies via hierarchical prosody models. In: CVPR. pp. 14687–14697 (2023)

12. Cong, G., Pan, J., Li, L., Qi, Y., Peng, Y., Hengel, A.v.d., Yang, J., Huang, Q.: Emodubber: Towards high quality and emotion controllable movie dubbing. arXiv preprint arXiv:2412.08988 (2024)

13. Cong, G., Qi, Y., Li, L., Beheshti, A., Zhang, Z., Hengel, A.v.d., Yang, M.H., Yan, C., Huang, Q.: Styledubber: Towards multi-scale style learning for movie dubbing. arXiv preprint arXiv:2402.12636 (2024)

14. Du, Z., Wang, Y., Chen, Q., Shi, X., Lv, X., Zhao, T., Gao, Z., Yang, Y., Gao, C., Wang, H., et al.: Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117 (2024)

15. Guo, Y., Du, C., Ma, Z., Chen, X., Yu, K.: Voiceflow: Efficient text-to-speech with rectified flow matching. In: ICASSP. pp. 11121–11125 (2024)

16. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)

17. Hu, C., Tian, Q., Li, T., Wang, Y., Wang, Y., Zhao, H.: Neural dubber: Dubbing for videos according to scripts. In: NeurIPS. pp. 16582–16595 (2021)

18. Jiang, Z., Ren, Y., Li, R., Ji, S., Zhang, B., Ye, Z., Zhang, C., Jionghao, B., Yang, X., Zuo, J., et al.: Megatts 3: Sparse alignment enhanced latent diffusion transformer for zero-shot speech synthesis. arXiv preprint arXiv:2502.18924 (2025)

19. Kim, J.H., Choi, J., Kim, J., Jung, C., Chung, J.S.: From faces to voices: Learning hierarchical representations for high-quality video-to-speech. arXiv preprint arXiv:2503.16956 (2025)

20. Kim, J., Kim, J., Chung, J.S.: Let there be sound: Reconstructing high quality speech from silent videos. In: AAAI. pp. 2759–2767 (2024)

21. Kim, M., Hong, J., Ro, Y.M.: Lip-to-speech synthesis in the wild with multi-task learning. In: ICASSP. pp. 1–5 (2023)

22. Kim, S., Shih, K.J., Badlani, R., Santos, J.F., Bakhturina, E., Desta, M., Valle, R., Yoon, S., Catanzaro, B.: P-flow: A fast and data-efficient zero-shot TTS through speech prompting. In: NeurIPS (2023)

23. Le, M., Vyas, A., Shi, B., Karrer, B., Sari, L., Moritz, R., Williamson, M., Manohar, V., Adi, Y., Mahadeokar, J., Hsu, W.: Voicebox: Text-guided multilingual universal speech generation at scale. In: NeurIPS (2023)

24. Li, L., Cong, G., Qi, Y., Zha, Z.J., Wu, Q., Sheng, Q.Z., Huang, Q., Yang, M.H.: Dubbing movies via hierarchical phoneme modeling and acoustic diffusion denoising. IEEE TPAMI 47(11), 10361–10377 (2025)

25. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

26. Liu, J., Xiang, Y., Zhao, H., Li, X., Ling, Z.: Funcineforge: A unified dataset toolkit and model for zero-shot movie dubbing in diverse cinematic scenes. arXiv preprint arXiv:2601.14777 (2026)

27. Liu, R., Zhao, Y., Jia, Z.: Towards authentic movie dubbing with retrieveaugmented director-actor interaction learning. arXiv preprint arXiv:2511.14249 (2025)

28. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

29. Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., Chen, X.: emotion2vec: Selfsupervised pre-training for speech emotion representation. In: Findings of ACL. pp. 15747–15760 (2024)

30. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., Sonderegger, M.: Montreal forced aligner: Trainable text-speech alignment using kaldi. In: Interspeech. pp. 498–502 (2017)

31. Mehta, S., Tu, R., Beskow, J., Székely, É., Henter, G.E.: Matcha-tts: A fast tts architecture with conditional flow matching. In: ICASSP. pp. 11341–11345 (2024)

32. Nguyen, N.S., Tran, T.V., Choi, J., Huynh-Nguyen, H.N., Hy, T.S., Nguyen, V.: Diflowdubber: Discrete flow matching for automated video dubbing via cross-modal alignment and synchronization. arXiv preprint arXiv:2603.14267 (2026)

33. Prajwal, K.R., Mukhopadhyay, R., Namboodiri, V.P., Jawahar, C.V.: Learning individual speaking styles for accurate lip to speech synthesis. In: CVPR. pp. 13793– 13802 (2020)

34. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision. In: ICML. pp. 28492–28518 (2023)

35. Rawal, R., Saifullah, K., Basri, R., Jacobs, D., Somepalli, G., Goldstein, T.: Cinepile: A long video question answering dataset and benchmark. In: CVPR Workshop (2024)

36. Reddy, C.K., Gopal, V., Cutler, R.: Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In: ICASSP. pp. 6493–6497. IEEE (2021)

37. Shi, B., Hsu, W., Lakhotia, K., Mohamed, A.: Learning audio-visual speech representation by masked multimodal cluster prediction. In: ICLR (2022)

38. Sung-Bin, K., Choi, J., Peng, P., Chung, J.S., Oh, T.H., Harwath, D.: Voicecraftdub: Automated video dubbing with neural codec language models. arXiv preprint arXiv:2504.02386 (2025)

39. Tian, W., Zhu, X., Liu, H., Zhao, Z., Chen, Z., Ding, C., Di, X., Zheng, J., Xie, L.: Dualdub: Video-to-soundtrack generation via joint speech and background audio synthesis. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 10671–10680 (2025)

40. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. Trans. Mach. Learn. Res. 2024 (2024)

41. Wan, L., Wang, Q., Papir, A., López-Moreno, I.: Generalized end-to-end loss for speaker verification. In: ICASSP. pp. 4879–4883 (2018)

42. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders. In: CVPR. pp. 16133–16142 (2023)

43. Yaman, D., Eyiokur, F.I., Bärmann, L., Akti, S., Ekenel, H.K., Waibel, A.: Audiovisual speech representation expert for enhanced talking face video generation and evaluation. In: CVPR Workshop. pp. 6003–6013 (2024)

44. Yao, J., Yang, Y., Pan, Y., Ning, Z., Ye, J., Zhou, H., Xie, L.: Stablevc: Style controllable zero-shot voice conversion with conditional flow matching. arXiv preprint arXiv:2412.04724 (2024)

45. Ye, J., Cao, B., Shan, H.: Emotional face-to-speech. In: Int. Conf. on Mach. Learn. vol. 267 (2025)

46. Ye, J., Cong, G., Wang, C., Wen, X.C., Li, Z., Cao, B., Shan, H.: Hierarchical codec diffusion for video-to-speech generation. In: IEEE Conf. Comput. Vis. Pattern Recog. (2026)

47. Ye, J., Zhang, J., Shan, H.: DepMamba: Progressive fusion mamba for multimodal depression detection. In: IEEE Conf. Acoust. Speech Signal Process. pp. 1–5 (2025)

48. Yu, J., Zhu, H., Jiang, L., Loy, C.C., Cai, W., Wu, W.: Celebv-text: A large-scale facial text-video dataset. In: CVPR. pp. 14805–14814 (2023)

49. Zhang, X., Wang, Y., Wang, C., Li, Z., Chen, Z., Wu, Z.: Advancing zero-shot text-to-speech intelligibility across diverse domains via preference alignment. In: ACL. pp. 12251–12270 (2025)

50. Zhang, Z., Li, L., Cong, G., Haibing, Y., Gao, Y., Yan, C., van den Hengel, A., Qi, Y.: From speaker to dubber: Movie dubbing with prosody and duration consistency learning. In: ACM MM (2024)

51. Zhang, Z., Li, L., Cong, G., Liu, C., Gao, Y., Wang, X., Gu, T., Qi, Y.: Instructdubber: Instruction-based alignment for zero-shot movie dubbing. arXiv preprint arXiv:2512.17154 (2025)

52. Zhang, Z., Li, L., Yan, C., Liu, C., van den Hengel, A., Qi, Y.: Prosody-enhanced acoustic pre-training and acoustic-disentangled prosody adapting for movie dubbing. arXiv preprint arXiv:2503.12042 (2025)

53. Zhao, Y., Jia, Z., Liu, R., Hu, D., Bao, F., Gao, G.: Mcdubber: Multimodal contextaware expressive video dubbing. arXiv preprint arXiv:2408.11593 (2024)

54. Zheng, J., Chen, Z., Ding, C., Di, X.: Deepdubber-v1: Towards high quality and dialogue, narration, monologue adaptive movie dubbing via multi-modal chain-ofthoughts reasoning guidance. arXiv preprint arXiv:2503.23660 (2025)

55. Zhou, S., Zhou, Y., He, Y., Zhou, X., Wang, J., Deng, W., Shu, J.: Indextts2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech. arXiv preprint arXiv:2506.21619 (2025)

56. Zhu, H., Wu, W., Zhu, W., Jiang, L., Tang, S., Zhang, L., Liu, Z., Loy, C.C.: Celebv-hq: A large-scale video facial attributes dataset. In: ECCV. pp. 650–667 (2022)