import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    video_source_arg = DeclareLaunchArgument(
            'video_source',
            default_value=TextSubstitution(text='video0'),
            # default_value=TextSubstitution(text='/home/dg/Videos/ts_20260108_143035_966288_fixed.mjpeg'),
            description='Video source file or device index for the drone tracker node.'
        )
    
    save_path_arg = DeclareLaunchArgument(
            'save_path',
            default_value='/home/dg/Videos/',
            description='Path to save videos.'
        )
    
    enable_save_arg = DeclareLaunchArgument(
            'enable_save',
            default_value='false',
            description='Whether to save videos.'
        )
    
    enable_visualize_arg = DeclareLaunchArgument(
            'enable_visualize',
            default_value='true',
            description='Whether to visualize results.'
        )

    enable_dcf_arg = DeclareLaunchArgument(
            'enable_dcf',
            default_value='false',
            description='Enable grayscale DCF fusion with NanoTrack (A/B testing).'
        )

    publish_prediction_arg = DeclareLaunchArgument(
        'publish_prediction', default_value='false',
        description='Extrapolate bbox/LOS to the newest fresh gimbal pose; false uses capture time.'
    )
    enable_preview_arg = DeclareLaunchArgument(
        'enable_preview', default_value='true',
        description='Publish live camera preview before ON and after OFF.'
    )
    los_rate_filter_tau_arg = DeclareLaunchArgument(
        'los_rate_filter_tau', default_value='0.12',
        description='LOS angular-rate low-pass time constant in seconds; 0 disables filtering.'
    )

    config = os.path.join(
        get_package_share_directory('drone_tracker'),
        'parameters',
        'balloon_tracker_params.yaml'
    )

    return LaunchDescription([
        video_source_arg,
        save_path_arg,
        enable_save_arg,
        enable_visualize_arg,
        enable_dcf_arg,
        publish_prediction_arg,
        enable_preview_arg,
        los_rate_filter_tau_arg,
        Node(
            package='drone_tracker',
            executable='drone_tracker_node',
            name='drone_tracker_node',
            parameters=[
                config,
                {'video_source': LaunchConfiguration('video_source')},
                {'save_path': LaunchConfiguration('save_path')},
                {'need_save': LaunchConfiguration('enable_save')},
                {'visualized_result': LaunchConfiguration('enable_visualize')},
                {'tracker.enable_dcf': LaunchConfiguration('enable_dcf')},
                {'publish_prediction': ParameterValue(LaunchConfiguration('publish_prediction'), value_type=bool)},
                {'enable_preview': ParameterValue(LaunchConfiguration('enable_preview'), value_type=bool)},
                {'los.rate_filter_tau': ParameterValue(LaunchConfiguration('los_rate_filter_tau'), value_type=float)}],
            output='screen',
            emulate_tty=True,
        )
    ])
