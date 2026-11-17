# Third-party components and assets

Model source and weights are not vendored. The installer fetches pinned upstream
checkouts into ignored `.third_party/` directories.

| Component | Upstream | License file |
|---|---|---|
| SAM2 | https://github.com/facebookresearch/sam2 | `third_party/licenses/SAM2-LICENSE` |
| SAM3 | https://github.com/facebookresearch/sam3 | `third_party/licenses/SAM3-LICENSE` |
| ProPainter | https://github.com/sczhou/ProPainter | `third_party/licenses/ProPainter-LICENSE` |

The ProPainter inference patch retains its upstream license. S-Lab License 1.0
contains non-commercial-use conditions; this adapter does not change them. SAM3
retains its applicable Meta license and checkpoint access requirements. Consult the
included license files and upstream weight terms.

`assets/robot` contains the MOZ/Wuji URDF and its referenced meshes.
`assets/tactile` contains the sensor layout, rotation header, and coordinate
spreadsheet. Third-party components and assets remain subject to their
respective license terms.
