"""Palm-rooted hand submodels extracted from the trusted full-body URDF."""

from __future__ import annotations

from retarget.robot.urdf_model import URDFModel


LEFT_TIP_LINKS = [f"left_finger{i}_tip_link" for i in range(1, 6)]
RIGHT_TIP_LINKS = [f"right_finger{i}_tip_link" for i in range(1, 6)]


def hand_submodel_from_full(
    full_model: URDFModel,
    root_link: str,
    tip_links: list[str],
    expected_dof: int = 20,
) -> URDFModel:
    """Extract one palm subtree without loading a second hand URDF."""
    descendants: list[str] = []
    stack = [root_link]
    while stack:
        link = stack.pop()
        if link in descendants:
            continue
        descendants.append(link)
        stack.extend(reversed(full_model.link_children.get(link, [])))
    descendant_set = set(descendants)
    all_joints = [
        joint
        for joint in full_model.all_joints
        if joint.parent_link in descendant_set and joint.child_link in descendant_set
    ]
    joints = [joint for joint in full_model.joints if joint in all_joints]
    selected_tips = [tip for tip in tip_links if tip in descendant_set]
    if len(joints) != expected_dof:
        raise ValueError(
            f"{root_link} subtree should contain {expected_dof} active joints, got {len(joints)}"
        )
    if len(selected_tips) != len(tip_links):
        raise ValueError(f"{root_link} is missing fingertip links: {set(tip_links) - set(selected_tips)}")
    return URDFModel(
        name=f"{full_model.name}_{root_link}_subtree",
        urdf_path=full_model.urdf_path,
        root_link=root_link,
        joints=joints,
        all_joints=all_joints,
        joint_order=[joint.name for joint in joints],
        link_names=descendants,
        link_visuals={link: full_model.link_visuals.get(link, []) for link in descendants},
        tip_links=selected_tips,
        link_children={
            link: [
                child
                for child in full_model.link_children.get(link, [])
                if child in descendant_set
            ]
            for link in descendants
        },
        child_to_joint={joint.child_link: joint for joint in joints},
        child_to_all_joint={joint.child_link: joint for joint in all_joints},
    )


def dual_hand_submodels_from_full(
    full_model: URDFModel,
    expected_dof_per_hand: int = 20,
) -> tuple[URDFModel, URDFModel]:
    return (
        hand_submodel_from_full(
            full_model,
            "left_palm_link",
            LEFT_TIP_LINKS,
            expected_dof_per_hand,
        ),
        hand_submodel_from_full(
            full_model,
            "right_palm_link",
            RIGHT_TIP_LINKS,
            expected_dof_per_hand,
        ),
    )
