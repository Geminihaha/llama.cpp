
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/private';
 * 
 * console.log(ENVIRONMENT); // => "production"
 * console.log(PUBLIC_BASE_URL); // => throws error during build
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/private' {
	export const TERMUX_APP__SE_INFO: string;
	export const DEX2OATBOOTCLASSPATH: string;
	export const TERMUX_APP__LEGACY_DATA_DIR: string;
	export const TERMUX_MAIN_PACKAGE_FORMAT: string;
	export const USER: string;
	export const SSH_CLIENT: string;
	export const npm_config_user_agent: string;
	export const EXTERNAL_STORAGE: string;
	export const SVDIR: string;
	export const npm_node_execpath: string;
	export const SHLVL: string;
	export const npm_config_noproxy: string;
	export const TERMUX__ROOTFS_DIR: string;
	export const HOME: string;
	export const OLDPWD: string;
	export const SSH_TTY: string;
	export const npm_package_json: string;
	export const TERMUX_APP__SE_FILE_CONTEXT: string;
	export const TERMUX_APP_PID: string;
	export const LC_MONETARY: string;
	export const npm_config_userconfig: string;
	export const npm_config_local_prefix: string;
	export const BOOTCLASSPATH: string;
	export const npm_config_engine_strict: string;
	export const COLOR: string;
	export const TMUX_TMPDIR: string;
	export const TMPDIR: string;
	export const TERMUX_APP__DATA_DIR: string;
	export const GTK_IM_MODULE: string;
	export const LOGNAME: string;
	export const TERMUX__HOME: string;
	export const _: string;
	export const npm_config_prefix: string;
	export const npm_config_npm_version: string;
	export const TERMUX_VERSION: string;
	export const TERM: string;
	export const npm_config_cache: string;
	export const ANDROID_DATA: string;
	export const HISTCONTROL: string;
	export const npm_config_node_gyp: string;
	export const PATH: string;
	export const TERMUX__SE_PROCESS_CONTEXT: string;
	export const NODE: string;
	export const npm_package_name: string;
	export const LC_ADDRESS: string;
	export const TERMUX_APK_RELEASE: string;
	export const ANDROID_I18N_ROOT: string;
	export const ANDROID_ROOT: string;
	export const LD_PRELOAD: string;
	export const LANG: string;
	export const TERMUX__PREFIX: string;
	export const LC_TELEPHONE: string;
	export const XDG_CONFIG_HOME: string;
	export const XMODIFIERS: string;
	export const npm_lifecycle_script: string;
	export const PREFIX: string;
	export const SHELL: string;
	export const LC_NAME: string;
	export const LOGDIR: string;
	export const ANDROID_TZDATA_ROOT: string;
	export const npm_package_version: string;
	export const npm_lifecycle_event: string;
	export const TERMUX_IS_DEBUGGABLE_BUILD: string;
	export const npm_config_foreground_scripts: string;
	export const LC_MEASUREMENT: string;
	export const LC_IDENTIFICATION: string;
	export const ANDROID_SDK_ROOT: string;
	export const QT_IM_MODULE: string;
	export const npm_config_globalconfig: string;
	export const npm_config_init_module: string;
	export const JAVA_HOME: string;
	export const PWD: string;
	export const LC_ALL: string;
	export const npm_execpath: string;
	export const SSH_CONNECTION: string;
	export const ANDROID_HOME: string;
	export const npm_config_global_prefix: string;
	export const ANDROID__BUILD_VERSION_SDK: string;
	export const npm_command: string;
	export const LC_PAPER: string;
	export const ANDROID_API_LEVEL: string;
	export const ANDROID_ART_ROOT: string;
	export const INIT_CWD: string;
	export const EDITOR: string;
}

/**
 * This module provides access to environment variables that are injected _statically_ into your bundle at build time and are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Static environment variables are [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env` at build time and then statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * For example, given the following build time environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { ENVIRONMENT, PUBLIC_BASE_URL } from '$env/static/public';
 * 
 * console.log(ENVIRONMENT); // => throws error during build
 * console.log(PUBLIC_BASE_URL); // => "http://site.com"
 * ```
 * 
 * The above values will be the same _even if_ different values for `ENVIRONMENT` or `PUBLIC_BASE_URL` are set at runtime, as they are statically replaced in your code with their build time values.
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are limited to _private_ access.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Private_ access:**
 * 
 * - This module cannot be imported into client-side code
 * - This module includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured)
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://site.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * 
 * console.log(env.ENVIRONMENT); // => "production"
 * console.log(env.PUBLIC_BASE_URL); // => undefined
 * ```
 */
declare module '$env/dynamic/private' {
	export const env: {
		TERMUX_APP__SE_INFO: string;
		DEX2OATBOOTCLASSPATH: string;
		TERMUX_APP__LEGACY_DATA_DIR: string;
		TERMUX_MAIN_PACKAGE_FORMAT: string;
		USER: string;
		SSH_CLIENT: string;
		npm_config_user_agent: string;
		EXTERNAL_STORAGE: string;
		SVDIR: string;
		npm_node_execpath: string;
		SHLVL: string;
		npm_config_noproxy: string;
		TERMUX__ROOTFS_DIR: string;
		HOME: string;
		OLDPWD: string;
		SSH_TTY: string;
		npm_package_json: string;
		TERMUX_APP__SE_FILE_CONTEXT: string;
		TERMUX_APP_PID: string;
		LC_MONETARY: string;
		npm_config_userconfig: string;
		npm_config_local_prefix: string;
		BOOTCLASSPATH: string;
		npm_config_engine_strict: string;
		COLOR: string;
		TMUX_TMPDIR: string;
		TMPDIR: string;
		TERMUX_APP__DATA_DIR: string;
		GTK_IM_MODULE: string;
		LOGNAME: string;
		TERMUX__HOME: string;
		_: string;
		npm_config_prefix: string;
		npm_config_npm_version: string;
		TERMUX_VERSION: string;
		TERM: string;
		npm_config_cache: string;
		ANDROID_DATA: string;
		HISTCONTROL: string;
		npm_config_node_gyp: string;
		PATH: string;
		TERMUX__SE_PROCESS_CONTEXT: string;
		NODE: string;
		npm_package_name: string;
		LC_ADDRESS: string;
		TERMUX_APK_RELEASE: string;
		ANDROID_I18N_ROOT: string;
		ANDROID_ROOT: string;
		LD_PRELOAD: string;
		LANG: string;
		TERMUX__PREFIX: string;
		LC_TELEPHONE: string;
		XDG_CONFIG_HOME: string;
		XMODIFIERS: string;
		npm_lifecycle_script: string;
		PREFIX: string;
		SHELL: string;
		LC_NAME: string;
		LOGDIR: string;
		ANDROID_TZDATA_ROOT: string;
		npm_package_version: string;
		npm_lifecycle_event: string;
		TERMUX_IS_DEBUGGABLE_BUILD: string;
		npm_config_foreground_scripts: string;
		LC_MEASUREMENT: string;
		LC_IDENTIFICATION: string;
		ANDROID_SDK_ROOT: string;
		QT_IM_MODULE: string;
		npm_config_globalconfig: string;
		npm_config_init_module: string;
		JAVA_HOME: string;
		PWD: string;
		LC_ALL: string;
		npm_execpath: string;
		SSH_CONNECTION: string;
		ANDROID_HOME: string;
		npm_config_global_prefix: string;
		ANDROID__BUILD_VERSION_SDK: string;
		npm_command: string;
		LC_PAPER: string;
		ANDROID_API_LEVEL: string;
		ANDROID_ART_ROOT: string;
		INIT_CWD: string;
		EDITOR: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * This module provides access to environment variables set _dynamically_ at runtime and that are _publicly_ accessible.
 * 
 * |         | Runtime                                                                    | Build time                                                               |
 * | ------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
 * | Private | [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private) | [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private) |
 * | Public  | [`$env/dynamic/public`](https://svelte.dev/docs/kit/$env-dynamic-public)   | [`$env/static/public`](https://svelte.dev/docs/kit/$env-static-public)   |
 * 
 * Dynamic environment variables are defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`.
 * 
 * **_Public_ access:**
 * 
 * - This module _can_ be imported into client-side code
 * - **Only** variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`) are included
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 * 
 * > [!NOTE] To get correct types, environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * >
 * > ```env
 * > MY_FEATURE_FLAG=
 * > ```
 * >
 * > You can override `.env` values from the command line like so:
 * >
 * > ```sh
 * > MY_FEATURE_FLAG="enabled" npm run dev
 * > ```
 * 
 * For example, given the following runtime environment:
 * 
 * ```env
 * ENVIRONMENT=production
 * PUBLIC_BASE_URL=http://example.com
 * ```
 * 
 * With the default `publicPrefix` and `privatePrefix`:
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.ENVIRONMENT); // => undefined, not public
 * console.log(env.PUBLIC_BASE_URL); // => "http://example.com"
 * ```
 * 
 * ```
 * 
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
