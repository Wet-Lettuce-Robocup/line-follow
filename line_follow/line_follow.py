from geometry_msgs.msg import Twist
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import Image


class LineFollowerNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('line_follower')
        self.vel_pub = None
        self.image_sub = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image_sub = self.create_subscription(Image, '/camera/image_raw',
                                                  self.image_callback, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        if self.vel_pub is not None:
            self.destroy_publisher(self.vel_pub)

        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, image):
        pass
