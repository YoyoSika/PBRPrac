// Unity built-in shader source. Copyright (c) 2016 Unity Technologies. MIT license (see license.txt)

Shader "YoyoPbrPrac"
{
	Properties{
		 _Color("Color",Color) = (1,1,1,0)
		 _MainTex("MainTex", 2D) = "white" {}
		 _Normal("Normal", 2D) = "bump" {}
		 _CubeMap("CubeMap", Cube) = "_Skybox" {}
		 _Control("R(metallic)G(AO)B(Rough)", 2D) = "white" {}
		 metalpower("metallic",Range(0,1)) = 1
		 smoothpower("smooth",Range(-3,1)) = 1
		 aopower("ao",Range(0,1)) = 1
	}
		SubShader
		 {
			LOD 300
			Blend SrcAlpha OneMinusSrcAlpha
			ZWrite on
			ZTest on

			Tags {
				"RenderType" = "Opaque"
			}

		  Pass {
		  Blend SrcAlpha OneMinusSrcAlpha

		  Cull Back
		  ZWrite on
		  ZTest on
			  Name "FORWARD"
			  Tags {
				  "LightMode" = "ForwardBase"
			  }
			  CGPROGRAM
			  #pragma vertex vert
			  #pragma fragment frag
			  #define UNITY_PASS_FORWARDBASE
			  #define UNITY_LIGHTMAP_DLDR_ENCODING
			  #include "UnityCG.cginc"
			  #include "AutoLight.cginc"
			  #include "Lighting.cginc"
			  #include "UnityStandardBRDF.cginc"

			  #pragma multi_compile_fwdbase_fullshadows
			   #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
			  #pragma multi_compile_fog
			  #pragma target 3.0
			  uniform sampler2D _MainTex; uniform fixed4 _MainTex_ST;
			  uniform sampler2D _Normal; uniform fixed4 _Normal_ST;
			  uniform samplerCUBE _CubeMap;
			  uniform sampler2D _Control; uniform fixed4 _Control_ST;
			  uniform fixed4 _Color;
			  uniform fixed smoothpower;
			  uniform fixed metalpower;

			  fixed aopower;

			  /////////////////
			  struct VertexInput {
				  float4 vertex : POSITION;
				  float3 normal : NORMAL;
				  float4 tangent : TANGENT;
				  half2 texcoord0 : TEXCOORD0;
			  };
			  struct VertexOutput {
				  float4 pos : SV_POSITION;
				  half2 uv0 : TEXCOORD0;
				  float4 posWorld : TEXCOORD1;
				  float3 normalDir : TEXCOORD2;
				  float2 scrollUv : TEXCOORD3;
				  float3 tangentDir : TEXCOORD4;
				  float3 bitangentDir : TEXCOORD5;
				  LIGHTING_COORDS(6,7)
				  UNITY_FOG_COORDS(8)
			  };
			  VertexOutput vert(VertexInput v) {
				  VertexOutput o = (VertexOutput)0;
				  o.uv0 =  TRANSFORM_TEX(v.texcoord0, _MainTex);
				  o.normalDir = UnityObjectToWorldNormal(v.normal);
				  o.tangentDir = normalize(mul(unity_ObjectToWorld, fixed4(v.tangent.xyz, 0.0)).xyz);
				  o.bitangentDir = normalize(cross(o.normalDir, o.tangentDir) * v.tangent.w);
				  o.posWorld = mul(unity_ObjectToWorld, v.vertex);
				  o.pos = UnityObjectToClipPos(v.vertex);
				  
				  UNITY_TRANSFER_FOG(o,o.pos);
				  TRANSFER_VERTEX_TO_FRAGMENT(o)//根据该pass处理的光源类型（ spot 或 point 或 directional ）来计算光源坐标的具体值，以及进行和 shadow 相关的计算等
				  return o;
			  }
			  float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
			  {
				  return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
			  }
			  float4 frag(VertexOutput i) : COLOR {

				  i.normalDir = normalize(i.normalDir);
				  fixed3x3 tangentTransform = fixed3x3(i.tangentDir, i.bitangentDir, i.normalDir);
				  //↑↑↑↑↑↑这里是做按行排列，所以是从 世界坐标系转法线坐标系的矩阵 ↑↑↑↑↑↑

				  fixed3 tangentNormal = UnpackNormal(tex2D(_Normal, i.uv0));
				  fixed3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
				  float3 worldNormal = normalize(mul(tangentNormal, tangentTransform));//左乘到世界坐标系
				  fixed3 viewReflectDirection = reflect(-viewDirection, worldNormal);//视角的对称方向，用来采CubeMap
				  fixed3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
				  fixed3 halfDirection = normalize(viewDirection + lightDirection);

				  fixed4 controlMap = tex2D(_Control, i.uv0);//r metallic , g ao , b roughness
				  fixed metalic = controlMap.r *metalpower ;
				  fixed roughness =  controlMap.b * (1-smoothpower);  
				  fixed ao = controlMap.g ;
				  
				  fixed4 mainColor = tex2D(_MainTex,i.uv0);
				  fixed3 albedo = mainColor.rgb;



				  float perceptualRoughness = sqrt(roughness);


				  float LdotN = max(0.001,dot(lightDirection, worldNormal));
				  float LdotH = dot(lightDirection, halfDirection);//max(0.001,dot(lightDirection, halfDirection));
				  float NdotV = dot(worldNormal, viewDirection);// max(0.001, dot(worldNormal, viewDirection));
				  float NdotH = max(0.001,dot(worldNormal, halfDirection));

				  //fresnel
				  float3 F0 = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metalic);
				  //unity_ColorSpaceDielectricSpec 是一个很暗的值,模拟金属的黯淡
				  float3 fresnel = F0 + (1 - F0)  * pow(1 - LdotH, 5);

				  //BRDF - diffuse
				  fixed kd = (1 - fresnel) * (1 - metalic);
				  fixed3 diffuse = kd * max(0, LdotN)*_LightColor0.rgb * albedo ;
				  diffuse = lerp(diffuse, diffuse*ao, aopower);
				  
				  //接受影子
				  fixed attenuation = LIGHT_ATTENUATION(i);
				  diffuse *= attenuation;

				  //BRDF - Cook-Torrance
				  //V 与 D
				  half V = SmithBeckmannVisibilityTerm(LdotN, NdotV, roughness );
				  half D = NDFBlinnPhongNormalizedTerm(NdotH, PerceptualRoughnessToSpecPower(perceptualRoughness));

				  float3 brdfSpecular = _LightColor0 * fresnel * V * D ;
				  float3 brdf = brdfSpecular + 4 * diffuse/ UNITY_PI;

				  //IBL - cubeMap
				  float3 ibl;

				  //ibl specular
				  fixed3 surfaceReduction = 1.0 - 0.28*roughness*perceptualRoughness;
				  fixed cubeMip = perceptualRoughnessToMipmapLevel(perceptualRoughness);// 根据粗糙度得到CubeMap的LOD级别，越粗糙，越不需要CubeMap的反射
				  fixed4 iblSpecular1 = texCUBElod(_CubeMap, fixed4(viewReflectDirection,cubeMip));//带LOD采CubeMap
				  float3 iblSpecular = DecodeHDR(iblSpecular1, unity_SpecCube0_HDR);

				  float oneMinusReflectivity = 1 - max(max(brdfSpecular.r, brdfSpecular.g), brdfSpecular.b);
				  float grazingTerm = saturate(1 - roughness + (1 - oneMinusReflectivity));

				  fixed3 iblSpecularResult = iblSpecular.rgb *  surfaceReduction  * FresnelLerp(F0, grazingTerm, NdotV);


				  //ibl diffuse

				  float3 Flast = fresnelSchlickRoughness(max(NdotV, 0.0), F0, roughness);
				  float kdLast = (1 - Flast) * (1 - metalic);

				  float3 ambient_contrib = ShadeSH9(float4(worldNormal, 1));
				  float3 ambient = 0.03 * albedo;
				  float3 iblDiffuse = max(half3(0, 0, 0), ambient + ambient_contrib);
				  float3 iblDiffuseResult = iblDiffuse * kdLast * albedo;

				  ibl = iblDiffuseResult + iblSpecularResult;
				  

				  fixed3 finalColor =  brdf + ibl;
				  //debug code
				  //finalColor = fixed3(LdotH, LdotH,LdotH);
				  fixed4 finalRGBA = fixed4(finalColor,mainColor.a)*_Color;
				  UNITY_APPLY_FOG(i.fogCoord, finalRGBA);

				  return finalRGBA;

				 }
				 ENDCG


			 }

		 }

		FallBack "Legacy Shaders/Diffuse"

}

