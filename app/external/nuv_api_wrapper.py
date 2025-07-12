"""Contains the nuv API wrapping functionality."""

# Importing Python libraries
import os
import json
import time
import requests

TOKEN_UPDATE_INTERVAL_IN_SECONDS = 120

class NuvAPIWrapper:
    """Base class responsible for establishing the communication between nuvBench
    and the infrastructure through high-level abstractions for Nuv's APIs."""

    # Origin node IP and credentials
    origin_ip = ""
    origin_username = ""
    origin_password = ""

    # Edge node IP and credentials
    edge_ip = ""

    # Base API information
    org_name = ""
    org_domain = ""
    org_id = ""
    org_username = ""
    org_password = ""
    tribe_id = ""

    # Authentication tokens
    refresh_token = ""
    refresh_token_updated_at = None
    access_token = ""
    access_token_updated_at = None
    manager_token = ""

    # Cameras and devices
    devices = []
    cameras = []

    # Origin session
    origin_session = None

    # Requests response times
    requests_response_times = []

    @classmethod
    def refresh_token_has_expired(cls, expiration_threshold: int = 120):
        """Checks if the refresh token has expired based on a given expiration threshold (in seconds).

        Args:
            expiration_threshold (int, optional): Expiration threshold (in seconds). Defaults to 120.

        Returns:
            token_has_expired (bool): Whether the refresh token has expired or not.
        """
        token_has_expired = (
            cls.refresh_token_updated_at is None or int(time.time()) - int(cls.refresh_token_updated_at) >= expiration_threshold
        )
        return token_has_expired

    @classmethod
    def access_token_has_expired(cls, expiration_threshold: int = 120):
        return cls.access_token_updated_at is None or int(time.time()) - int(cls.access_token_updated_at) >= expiration_threshold

    @classmethod
    def run_request(cls, method_name: str, method_parameters: dict = {}):
        method_to_run = getattr(cls, method_name)

        # Updating tokens if necessary
        if method_name != "get_refresh_token" and method_name != "get_access_token":
            tokens_are_not_defined = cls.refresh_token == "" or cls.access_token == ""
            if tokens_are_not_defined or cls.refresh_token_has_expired() or cls.access_token_has_expired():
                cls.run_request(method_name="get_refresh_token")
                cls.run_request(method_name="get_access_token")

        # Executing the request and storing how long it took to return
        initial_time = time.time()
        method_output = method_to_run(parameters=method_parameters)
        final_time = time.time()
        cls.requests_response_times.append(
            {
                "method_name": method_name,
                "executed_at": initial_time,
                "response_time": final_time - initial_time,
                "method_output": method_output,
            }
        )

        #print(f"{json.dumps(cls.requests_response_times[-1], indent=4)}")

        return method_output

    @classmethod
    def get_refresh_token(cls, parameters: dict = {}) -> str:
        """Gets the nuv API refresh token.

        Returns:
            cls.refresh_token (str): Refresh token.
        """
        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7003/api/nvr-aa/v1/login/refresh-token?org={cls.org_domain}"

        # Defining the request content
        payload = f"username={cls.org_username}&password={cls.org_password}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Executing the request
        while True:
            response = requests.request("POST", url, headers=headers, data=payload)
            try:
                response = response.json()
                cls.refresh_token = response["refresh_token"]
                break
            except:
                print("Fail while getting refresh token, trying again")
                time.sleep(1)

        # Updating the NuvAPIWrapper attributes according to the request output
        cls.refresh_token_updated_at = time.time()

        return cls.refresh_token

    @classmethod
    def get_access_token(cls, parameters: dict = {}) -> str:
        """Gets the nuv API access token.

        Returns:
            cls.access_token (str): Access token.
        """
        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7003/api/nvr-aa/v1/access-token/"

        # Defining the request content
        payload = {}
        headers = {"Authorization": f"Bearer {cls.refresh_token}"}

        # Executing the request
        response = requests.request("POST", url, headers=headers, data=payload)

        # Updating the NuvAPIWrapper attributes according to the request output
        cls.access_token = response.json()["access_token"]
        cls.access_token_updated_at = time.time()

        return cls.access_token


    @classmethod
    def get_manager_token(cls, parameters: dict = {}) -> str:
        """Gets the manager token from the Origin host '.env' file.

        Returns:
            cls.manager_token (str): Manager token.
        """
        # Gathering the external '.env' file from the Origin host
        os.system(
            f"sshpass -p '{cls.origin_password}' scp -o StrictHostKeyChecking=accept-new {cls.origin_username}@{cls.origin_ip}:/usr/local/etc/nvr-origin/.env ./.env.nuv"
        )

        # Parsing the '.env' file to get the manager token
        with open(".env.nuv") as file:
            for line in file:
                if "MANAGER_TOKEN" in line:
                    cls.manager_token = line.split("=")[1].strip()
                    break

        # Removing the '.env' file
        os.system("rm .env.nuv")

        return cls.manager_token



    # @classmethod
    # def get_manager_token(cls, parameters: dict = {}) -> str:
    #     """Gets the manager token from the Origin host '.env' file.
    #
    #     Returns:
    #         cls.manager_token (str): Manager token.
    #     """
    #
    #     env_path = "/usr/local/etc/nvr-origin/.env" if "custom_path" not in parameters else parameters["custom_path"]
    #
    #
    #     # Parsing the '.env' file to get the manager token
    #     print(env_path,flush=True)
    #     with open(env_path) as file:
    #         for line in file:
    #             if "MANAGER_TOKEN" in line:
    #                 cls.manager_token = line.split("=")[1].strip()
    #                 break
    #
    #     return cls.manager_token

    @classmethod
    def get_org_id(cls, parameters: dict = {}) -> int:
        """Gets the organization ID.

        Returns:
            cls.org_id (str): Organization ID.
        """
        # Defining the base request endpoint
        url = f"http://{cls.edge_ip}:7005/api/nvr-edge/v1/organizations/my/"

        # Defining the request content
        payload = {}
        headers = {"Authorization": f"Bearer {cls.access_token}"}

        # Executing the request
        response = requests.request("GET", url, headers=headers, data=payload)

        # If there are multiple Orgs registered in Nuv: Filtering the request output
        # to get the ID of the desired Org (whose name was passed in the input file)
        if type(response.json()) is list:
            for org in response.json():
                if org["name"] == cls.org_name:
                    # Updating the NuvAPIWrapper "org_id" attribute according to the request output
                    cls.org_id = org["id"]
                    break
        else:
            # Updating the NuvAPIWrapper "org_id" attribute according to the request output
            cls.org_id = response.json()["id"]

        return cls.org_id

    @classmethod
    def get_tribe_id(cls, parameters: dict = {}) -> int:
        """Gets the tribe ID.

        Returns:
            cls.tribe_id (str): Tribe ID.
        """
        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/organizations/{cls.org_id}/tribes/manager/"

        # Defining the request content
        payload = {}
        headers = {"manager-token": f"{cls.manager_token}"}

        # Executing the request
        response = requests.request("GET", url, headers=headers, data=payload)

        # If there are multiple tribes registered in Nuv: Filtering the request output to get
        # the ID of the tribe from the desired Org (whose name was passed in the input file)
        if type(response.json()) is list:
            for tribe in response.json():
                if tribe["organization_id"] == cls.org_id:
                    # Updating the NuvAPIWrapper "tribe_id" attribute according to the request output
                    cls.tribe_id = tribe["id"]
                    break
        else:
            # Updating the NuvAPIWrapper "tribe_id" attribute according to the request output
            print(response.json())
            input()
            cls.tribe_id = response.json()["id"]

        return cls.tribe_id

    @classmethod
    def get_all_devices(cls, parameters: dict = {}) -> list:
        """Gets the metadata of registered devices.

        Returns:
            cls.devices (list): Metadata of registered devices.
        """
        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/tribes/{cls.tribe_id}/devices/manager/"

        # Defining the request content
        payload = {}
        headers = {"manager-token": f"{cls.manager_token}"}

        # Executing the request
        response = requests.request("GET", url, headers=headers, data=payload)

        # Updating the NuvAPIWrapper "devices" attribute according to the request output
        cls.devices = response.json()

        return cls.devices

    @classmethod
    def create_device(cls, parameters: dict = {}) -> object:
        """Registers a device inside Nuv's database.

        Returns:
            device (dict): Registered device.
        """
        name = parameters["name"] if "name" in parameters else None
        user = parameters["user"] if "user" in parameters else None
        owner_tribe_id = parameters["owner_tribe_id"] if "owner_tribe_id" in parameters else None
        model_id = parameters["model_id"] if "model_id" in parameters else 29
        ddns = parameters["ddns"] if "ddns" in parameters else None
        host = parameters["host"] if "host" in parameters else None
        host_v6 = parameters["host_v6"] if "host_v6" in parameters else None
        port = parameters["port"] if "port" in parameters else None
        longitude = parameters["longitude"] if "longitude" in parameters else 0
        latitude = parameters["latitude"] if "latitude" in parameters else 0
        is_active = parameters["is_active"] if "is_active" in parameters else True

        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/tribes/1/devices/manager/"

        # Defining the request content
        payload = json.dumps(
            {
                "name": name,
                "user": user,
                "owner_tribe_id": owner_tribe_id,
                "model_id": model_id,
                "ddns": ddns,
                "host": host,
                "host_v6": host_v6,
                "port": port,
                "longitude": longitude,
                "latitude": latitude,
                "is_active": is_active,
            }
        )
        headers = {"manager-token": f"{cls.manager_token}", "Content-Type": "application/json"}

        # Executing the request
        response = requests.request("POST", url, headers=headers, data=payload)

        # Updating the NuvAPIWrapper "devices" attribute according to the request output
        device = response.json()
        cls.devices.append(device)

        return device

    @classmethod
    def remove_device(cls, parameters: dict = {}) -> None:
        """Removes a device from Nuv's database."""
        device_id = parameters["device_id"] if "device_id" in parameters else None

        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/tribes/{cls.tribe_id}/devices/manager/{device_id}?force=true"

        # Defining the request content
        payload = {}
        headers = {"manager-token": f"{cls.manager_token}"}

        # Executing the request
        requests.request("DELETE", url, headers=headers, data=payload)

    @classmethod
    def create_camera(cls, parameters: dict = {}) -> object:
        """Registers a device inside Nuv's database.

        Returns:
            device (dict): Registered device.
        """
        number = parameters["number"] if "number" in parameters else None
        rtsp = parameters["rtsp"] if "rtsp" in parameters else None
        device_id = parameters["device_id"] if "device_id" in parameters else None
        is_rtmp = parameters["is_rtmp"] if "is_rtmp" in parameters else False
        is_active = parameters["is_active"] if "is_active" in parameters else True
        quality = parameters["quality"] if "quality" in parameters else "sd"
        subscription_id = parameters["subscription_id"] if "subscription_id" in parameters else 1
        just_admins_can_access = parameters["just_admins_can_access"] if "just_admins_can_access" in parameters else False
        title = parameters["title"] if "title" in parameters else None

        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/devices/cameras/manager/"

        # Defining the request content
        payload = json.dumps(
            {
                "number": number,
                "rtsp": rtsp,
                "device_id": device_id,
                "is_rtmp": is_rtmp,
                "is_active": is_active,
                "quality": quality,
                "subscription_id": subscription_id,
                "just_admins_can_access": just_admins_can_access,
                "title": title,
            }
        )
        headers = {"manager-token": f"{cls.manager_token}", "Content-Type": "application/json"}

        # Executing the request
        response = requests.request("POST", url, headers=headers, data=payload)

        # Updating the NuvAPIWrapper "devices" attribute according to the request output
        device = response.json()
        cls.devices.append(device)

        return device

    @classmethod
    def get_camera(cls, parameters: dict = {}) -> list:
        """Gets the metadata of an specific camera.

        Returns:
            camera (dict): Metadata of specific camera.
        """
        # Defining the base request endpoint
        url = f"http://{cls.origin_ip}:7000/api/nvr-origin/v1/devices/cameras/manager/{parameters['camera_id']}"

        # Defining the request content
        payload = {}
        headers = {"manager-token": f"{cls.manager_token}"}

        # Executing the request
        response = requests.request("GET", url, headers=headers, data=payload)

        return response.json()
