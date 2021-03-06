// Copyright (c) 2014 Liang Kun. All Rights Reserved.
// Authors: Liang Kun <liangkun@data-intelli.com>

name := "sann"

description := "Simple Artificial Neural Network"

organization := "org.sann"

version := "0.1"

scalaVersion := "2.11.4"

scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature", "-optimise")

libraryDependencies ++= {
  val breezeVersion = "0.11-SNAPSHOT"
  Seq(
    "org.scalanlp" %% "breeze" % breezeVersion,
    "org.scalanlp" %% "breeze-natives" % breezeVersion,
    "org.scalanlp" %% "breeze-viz" % "0.9",
    "org.scalatest" %% "scalatest" % "2.2.1" % "test"
  )
}

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

mainClass in assembly := Some("org.sann.Main")

assemblyOutputPath in assembly := file(s"${target.value}/sann-${version.value}.jar")
