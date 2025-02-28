from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    person_name_arg = DeclareLaunchArgument(
        'person_name', default_value='Default_Name',
        description='ชื่อของบุคคลที่ต้องการส่งให้โหนด main'
    )

    node1 = Node(
        package="follow_person",
        executable="main",
        output="screen",
        parameters=[{"person_name": LaunchConfiguration('person_name')}]
    )

    node2 = Node(
        package="follow_person",
        executable="image_processor"
    )

    node3 = Node(
        package="follow_person",
        executable="robot_control"
    )

    ld = LaunchDescription()
    ld.add_action(person_name_arg)  
    ld.add_action(node1)
    ld.add_action(node2)
    ld.add_action(node3)

    return ld
