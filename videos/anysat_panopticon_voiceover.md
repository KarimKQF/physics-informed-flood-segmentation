# AnySat and Panopticon Voiceover

Target duration: 8 to 12 minutes.

This voiceover is written to match `videos/anysat_panopticon_explanation.py`.
It is intentionally pedagogical rather than paper-review dense. The central
project invariant is repeated several times: DEM is not a model input.

## Section 0 - Opening

We begin with our actual flood-segmentation problem.

Our model sees a fifteen-channel satellite tensor: thirteen Sentinel-2 optical
and multispectral bands, plus two Sentinel-1 SAR channels. The target is a
binary semantic mask: background or non-water is class zero, water or flood is
class one, and invalid pixels are ignored with index minus one.

So the input is not an RGB image. It is more like:

`X in R^{15 x 512 x 512}`.

This matters because many recent segmentation models are inherited from natural
image vision. They expect three color channels. But our channels are physical
measurements from different sensors. If we squeeze everything into fake RGB, we
may throw away exactly the information that makes Earth Observation useful.

That is why AnySat and Panopticon are interesting here.

## Section 1 - Why RGB Is A Problem

For a classical computer-vision backbone, the input story is simple:

RGB image, three channels, standard vision backbone.

For our satellite data, the story is different:

Sentinel-2 bands plus Sentinel-1 SAR, fifteen channels, not RGB.

A satellite band is not just a color channel. Sentinel-2 bands sample different
parts of the spectrum. Some are visible bands, some are near infrared, and some
are short-wave infrared. Sentinel-1 SAR is active radar: it measures backscatter,
not reflected sunlight.

There are three common shortcuts.

First, select only RGB-like bands. That is easy, but it discards multispectral
and SAR information.

Second, project fifteen channels down to three. That is compact, but it forces a
learned compression before the model has a chance to reason over sensor physics.

Third, inflate a first convolution from three to fifteen input channels. This is
practical and is what many dense segmentation experiments do, but it does not
make the model intrinsically sensor-aware.

AnySat and Panopticon are attractive because they are designed for heterogeneous
Earth Observation data rather than retrofitted from ordinary RGB images.

## Section 2 - AnySat Intuition

AnySat is an Earth Observation model built for many resolutions, scales, and
modalities.

The key idea is not: one model for one fixed sensor.

The key idea is: one model that can learn across heterogeneous EO data.

Inputs may come from Sentinel-2, Sentinel-1, aerial imagery, Landsat, MODIS, or
other sensors. AnySat maps these different inputs into a shared Earth
Observation representation.

From that shared representation, downstream tasks can branch out: land cover,
crop classification, change detection, flood segmentation, burn scar mapping, or
deforestation monitoring.

For our project, this is the right kind of generality. We do not need a model
that knows only floods. We need a model that understands satellite modalities
well enough that the flood head is not fighting the input format.

## Section 3 - AnySat Architecture Intuition

The architectural intuition behind AnySat is JEPA-like learning.

JEPA means Joint Embedding Predictive Architecture. The model does not need to
reconstruct missing pixels exactly. Instead, it learns to predict target
representations in feature space.

One view of an Earth Observation tile becomes the context. The encoder maps that
context into a representation. Another view, or a masked target region, produces
a target representation. The model learns by making the predicted embedding
close to the target embedding.

This is important for EO because raw pixel reconstruction can be messy across
sensors. A SAR signal, an optical signal, and a low-resolution MODIS signal do
not have the same pixel semantics. Feature-space prediction is a cleaner
self-supervised objective.

AnySat also uses scale-adaptive spatial encoders. Different sensors observe the
Earth at different resolutions and spatial supports. Instead of pretending that
every sensor is the same camera, AnySat adapts the spatial encoding so a single
model can operate across diverse EO inputs.

## Section 4 - AnySat For Our Project

In our project, AnySat would replace the backbone, not the scientific protocol.

The input path would be:

Sentinel-2 optical bands plus Sentinel-1 SAR channels, into an AnySat wrapper,
then a segmentation head, then dense logits with shape `[B, 2, H, W]`.

After logits, the pipeline is the same:

Logits go through softmax to produce water probability. The water probability
and ground-truth mask feed CE or Dice+CE. The water probability and the DEM feed
the topographic loss and physical consistency metrics.

The DEM is not passed into AnySat.

That invariant must remain untouched. The DEM is only a loss-side and
metric-side signal.

The benefit is clear: AnySat is more natural than RGB-only backbones for
multimodal EO. It lets us test whether the topographic loss behaves differently
when the representation is learned from EO modalities.

The risks are also clear. We must audit the exact Sentinel-2 band expectations.
We may need a wrapper for our thirteen Sentinel-2 bands and two Sentinel-1
channels. We must check whether the model expects dates, time series, or
specific metadata. And for our protocol, it must produce dense segmentation
logits cleanly.

## Section 5 - Panopticon Intuition

Panopticon takes another powerful route.

It is an any-sensor Earth Observation foundation model based on the DINOv2
family of ideas.

The central insight is that the model should know what each channel physically
means.

An optical blue channel, a near-infrared channel, a short-wave infrared channel,
and a SAR VV channel should not be treated as anonymous slots in a tensor.

Each channel has an identity. Sentinel-2 channels can be described by
wavelength. Sentinel-1 channels can be described by radar mode, such as VV or
VH.

Panopticon uses these channel identities so the patch embedding can be
sensor-aware.

## Section 6 - Panopticon Architecture Intuition

A standard DINOv2-style image model starts with RGB patches. Those patches go
through a patch embedding and then into a Vision Transformer.

Panopticon modifies this idea for arbitrary Earth Observation sensors.

The model receives arbitrary channels plus channel identity information. Before
forming the final patch tokens, it uses cross-attention over channels. This lets
the model fuse different optical and SAR measurements flexibly.

Three ideas matter most.

First, different sensors observing the same geographic footprint can be treated
as different views of the same object.

Second, channel subsampling teaches the model to be robust when some channels
are missing or when the spectral configuration changes.

Third, cross-attention over channels gives the model a principled way to fuse
arbitrary sensor channels instead of assuming a fixed RGB input.

## Section 7 - Panopticon For Our Project

For our project, Panopticon is especially interesting because the input problem
is exactly a channel problem.

We have thirteen Sentinel-2 bands, each with spectral meaning, plus two
Sentinel-1 SAR channels, VV and VH.

A Panopticon integration would map those channels and identities into a
Panopticon backbone. The backbone would produce patch features. Then we would
attach or adapt a dense decoder so the final output is logits `[B, 2, H, W]`.

The benefits are strong. We avoid RGB compression. We get arbitrary-channel
handling. We get a DINOv2-like foundation-model logic adapted to Earth
Observation. And we can test whether sensor-aware features interact differently
with the topographic loss.

The risks are practical. Panopticon is not automatically our full segmentation
pipeline. We need a dense head. Patch resolution may require upsampling. The
integration is likely more complex than SegMAN. And again, the DEM must stay out
of the input.

## Section 8 - Comparison

The four model families have different roles.

AnySat is a multimodal EO model for many resolutions, scales, and modalities.
Its strength is broad EO downstream use, including segmentation-like tasks. Its
risk is input formatting and band mapping. Its role is an EO multimodal SOTA
candidate after TerraMind.

Panopticon is an any-sensor DINOv2-like EO foundation model. Its strength is
arbitrary optical and SAR channel handling. Its risk is that we need a dense
segmentation decoder. Its role is perhaps the best conceptual match for our
fifteen-channel input problem.

SegMAN is a dense segmentation model. Its strength is that it is already
integrated and directly emits logits. Its risk is that it is not EO-native. Its
role is our current validated baseline.

TerraMind is an Earth Observation foundation-model baseline. Its strength is
domain alignment. Its risk is a heavier TerraTorch pipeline. Its role is
continuity with the earlier phase of the project.

## Section 9 - Physics-Informed Pipeline

The backbone can change, but the scientific protocol should stay fixed.

Any model must eventually produce dense logits `[B, 2, H, W]`. The logits go
through softmax and produce water probability.

Then we compare the same four losses:

CE.

Dice plus CE.

Dice plus CE plus topographic loss with the real DEM.

Dice plus CE plus topographic loss with shuffled DEM.

The DEM-shuffled control is not optional. It is how we separate real physical
information from a generic regularization effect.

This matters because in the current SegMAN low-data experiments, real DEM and
shuffled DEM behave similarly. That does not kill the idea. It tells us that the
next question is whether the same behavior appears on EO-native, multimodal, and
sensor-aware foundation models.

## Section 10 - Final Roadmap

The roadmap is conservative.

First, finish the SegMAN N=100 diagnostic.

Second, complete the TerraMind baseline.

Third, audit AnySat.

Fourth, audit Panopticon.

Fifth, integrate only if the model can produce dense logits cleanly.

Sixth, compare real DEM versus shuffled DEM on every backbone.

The final idea is simple:

AnySat and Panopticon are not general medical-image models. They are general
Earth Observation foundation models.

That is exactly the right level of generality for this project: not RGB-only,
not flood-only, but multimodal and sensor-aware.

