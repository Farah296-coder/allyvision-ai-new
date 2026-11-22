# AllyVision annotation guide

## Capture and privacy

- Use the final assistive-device camera position, height, pitch, lens/FOV, resolution and stabilization. Record these in the manifest; avoid mixing handheld chest-level and vehicle-mounted footage without a `camera_setup` tag.
- Capture varied Egyptian street scenes: busy and quiet streets, sidewalks, parked cars, motorcycles and tuk-tuks, pedestrians, street vendors/carts, poles, barriers, uneven paving, potholes, stairs, curbs, crossings, day/night and weather.
- Capture short sequences and keep every frame's `sequence_id` stable. Sample frames to reduce near duplicates; split whole sequences, not individual frames.
- Obtain required permissions/consent. Do not deliberately target identifiable people. Blur faces and license plates before annotation/sharing when appropriate; restrict access to raw footage and avoid publishing exact home/work locations.
- Record capture exclusions, privacy review, collection consent basis and redistribution permission. Do not infer a dataset license from the annotation tool.

## Object detection (boxes; independent taxonomy)

Use `config/detection_classes.yaml`, IDs 0–19. Draw a tight axis-aligned box around the visible object extent. Annotate each visible instance; do not label an empty road/sidewalk as an object. Mark truncated/occluded objects if their identity is still clear; skip identity-ambiguous regions rather than guessing. For crowds, annotate separable people; use the tool's ignore/crowd feature where supported and document its export handling.

- `person`: pedestrians, including vendors when a person is visible. A vendor's cart is `street_vendor_cart`; do not turn the person into a cart.
- `auto_rickshaw`: three-wheel motorized passenger/cargo rickshaw, including tuk-tuk. Keep separate from `motorcycle` and `car`.
- `traffic_sign` and `traffic_light`: each physical sign/signal unit; avoid boxes on text or light glow alone.
- `pole`, `fence`, `barrier`: label the physical obstacle when visible. Long continuous fence/barrier may be one box per visually separable segment; be consistent within a sequence.
- `animal`: visible non-human animal instances.
- `construction_obstacle`: temporary cones, debris, equipment or works that obstruct passage; do not use for permanent buildings or ordinary street furniture.
- `pothole`: visible road/sidewalk depression or broken surface that is clearly a hazard. Detection box only says where; severity/depth is not implied.

## Semantic segmentation (per-pixel surface/region taxonomy)

Use `config/segmentation_classes.yaml`. This is not the detection class list: one mask represents semantic regions; individual people/vehicles need not have separate instance IDs. Masks must be single-channel indexed PNGs (0–17); use 255 for unknown, unlabeled, ambiguous, or out-of-image pixels. Trace visible boundaries carefully; preserve thin curbs/poles where resolvable. Do not paint hidden surfaces behind objects.

- `sidewalk`: constructed pedestrian sidewalk surface, including uneven sidewalk. `walkable_ground`: clearly traversable pedestrian ground not otherwise sidewalk, such as a pedestrian plaza/path. Do not label a traffic lane as walkable merely because it is unobstructed.
- `road`: vehicle roadway. `crosswalk`: marked pedestrian crossing surface, not the whole intersection.
- `curb`: visible curb/step-edge band separating surface levels. `stairs`: stair treads/landings. `pothole`: visible damaged/depressed surface region; annotate only visible extent, not assumed depth.
- `building`, `wall`, `fence`, `vegetation`, `person`, `vehicle`, `traffic_sign`, `traffic_light`, `pole`: visible semantic pixels of the named region. `vehicle` includes cars, buses, trucks, motorcycles, bicycles and tuk-tuks.
- `other`: visible relevant scene content outside the current taxonomy. Use 255 instead for uncertainty or absent annotation; do not use `other` to guess traversability.

## Quality review

Review 100% of masks for ID range, dimensions and alignment; review a sample of boxes with overlays. Second-review all rare hazards and all `pothole`, `curb`, `stairs`, `auto_rickshaw` instances. Resolve class disagreements before training. Track per-class instance/image counts and labeler/version.

## CVAT / Roboflow setup

Create two independent projects/tasks, one for **object detection boxes** and one for **semantic masks**. Import the relevant taxonomy from the YAML files (copy IDs/names exactly). CVAT Community is MIT-licensed and supports Ultralytics YOLO box and polygon/mask exports plus semantic segmentation mask exports. It can be self-hosted for stronger control of raw footage; a hosted CVAT service still has its own service terms.

Roboflow supports hosted annotation, dataset versions, augmentations, and multiple downloadable formats; its current pricing page says projects are private by default unless published, with an opt-out for product-improvement/data licensing. Its Terms also grant broad rights for service/internal business purposes to content uploaded to private projects. Review the actual account settings, contract, retention and privacy policy before uploading identifiable footage; keep raw Egyptian captures in self-hosted CVAT if those terms are not acceptable. Export detection as Ultralytics YOLO. Semantic exports need explicit check/conversion to indexed PNG IDs; don't assume a color/palette mask stores target IDs correctly. Verify a round-trip sample and class IDs before bulk export. Neither tool's export format changes source dataset rights.
